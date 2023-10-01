"""An agent pool is a database of agent configs and their performance scores. As agents are trained over generations, runners will query the agent pool to select the best performing agent and mutate from them. 

We'll randomly select 1 of the N "high performers". Furthermore, we'll cap the number of available configs to select from based on the max population size.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Union
from pathlib import Path
from prodict import Prodict
from collections import deque
import bisect
import random
import socket
import time
import struct
import fcntl
import multiprocessing
import os
import pickle

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from cambrian.reinforce.evo import Agent

Performance = int


def AgentPoolFactory(config: Prodict, rank: int, *args, **kwargs):
    agent_pool_type = config.evo_config.agent_pool_type
    if agent_pool_type == "SharedFileAgentPool":
        return SharedFileAgentPool(config, rank, *args, **kwargs)
    if agent_pool_type == "SocketAgentPool":
        return SocketAgentPool(config, rank, *args, **kwargs)
    else:
        raise NotImplementedError


class AgentPool(ABC):
    """The agent pool base class. Throughout training, the agent pool is continuously updated to provide a realtime feed of model performance."""

    def __init__(self, config: Prodict, rank: int, *, verbose: int = 0):
        self.config = config
        self.rank = rank
        self.verbose = verbose
        self.population_size = self.config.evo_config.population_size
        self.num_top_performers = self.config.evo_config.num_top_performers
        self.pool = deque(maxlen=self.population_size + 1)

    def get_new_agent_config(self) -> Prodict:
        pool = self.get_pool()
        top_performer_pool = pool[: min(len(pool), self.num_top_performers)]
        return Prodict.from_dict(random.choice(top_performer_pool)[1])

    def insert_into_pool(self, performance: Performance, config: Prodict):
        position = bisect.bisect_left([perf for perf, _ in self.pool], performance)
        if self.verbose > 1:
            print(f"{self.rank}: Performance {performance:0.2f} received. Current pool: {[p for p,_ in self.pool]}. Inserting into pool at position {position}.")
        self.pool.insert(position, (performance, config))
        if len(self.pool) == self.population_size + 1:
            self.pool.popleft()

    @abstractmethod
    def write_to_pool(self, performance: Performance, config: Prodict):
        pass

    def get_pool(self) -> List[Tuple[Performance, Prodict]]:
        """Get the agent pool. This will be overridden based on the different storage mechanism.

        Returns:
            List[Tuple[Performance, Prodict]]: A **sorted** list from worst to best of agent configs in terms of performance
        """
        return list(self.pool)

    def close(self):
        pass


class SocketAgentPool(AgentPool):
    """This AgentPool implements the interface mechanism as a socket."""

    class Client:
        """Describes an endpoint/rank/client in which we can send data to/will send data to us."""

        def __init__(self, ip: str, port: int, rank: int):
            self.ip = ip
            self.port = port
            self.rank = rank

        def send(self, data: bytes, attempts: int = 10) -> bool:
            """Send data to this client."""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                for _ in range(attempts):
                    try:
                        s.connect((self.ip, self.port))
                        s.sendall(struct.pack("I", len(data)))
                        s.sendall(data)
                        return True
                    except ConnectionRefusedError:
                        time.sleep(2)

            return False

        def connect(self):
            """Just connect to server."""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.ip, self.port))

        def __str__(self) -> str:
            return f"Client(ip={self.ip}, port={self.port}, rank={self.rank})"

    class Server:
        """Helper class to interface with the `socket` library as a server. Each rank will have a server running in a separate thread (thread, not process, since we already have a lot of separate training processes)."""

        def __init__(self, ip: str, port: int):
            self.ip = ip
            self.port = port

        def start(self, num_clients: int):
            """Simply start listening. Non-blocking."""
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self.socket.bind((self.ip, self.port))
            self.socket.listen(num_clients)

        def accept(self) -> Tuple[socket.socket, Tuple[str, int]]:
            """Blocking; will wait for a connection."""
            return self.socket.accept()

        def receive(self, conn: socket.socket) -> bytes:
            """Will receive data from the specified connection. Data comes in two parts: the first 4 bytes is the length of the data, the rest is the data itself."""
            with conn:
                msg_length = conn.recv(4)
                if not msg_length:
                    raise Exception("No msg length received.")
                msg_length = struct.unpack("I", msg_length)[0]

                data = conn.recv(msg_length)
                if not data:
                    raise Exception("No data received.")

                return data

    def __init__(self, config: Prodict, rank: int, *, verbose: int = 0):
        super().__init__(config, rank, verbose=verbose)

        self.clients: List[self.Client] = []
        self._find_clients(config.evo_config.agent_pool_socket_port, Path(self.config.env_config.logdir) / self.config.env_config.exp_name / self.config.evo_config.agent_pool_socket_ip_folder)

        self.stop = False

        self.lock = multiprocessing.Lock()
        self.thread = multiprocessing.Process(
            target=self._start_server,
            args=(self.me.ip, config.evo_config.agent_pool_socket_port),
        )
        self.thread.start()

    def insert_into_pool(self, performance: Performance, config: Prodict):
        with self.lock:
            super().insert_into_pool(performance, config)

    def write_to_pool(self, performance: Performance, config: Prodict):
        data = pickle.dumps((performance, config))

        for client in self.clients:
            assert client.rank != self.rank

            success = client.send(data)
            if not success:
                print(f"WARNING: Failed to send data to client: {client}. Skipping...")

        # Finally, add it to our pool
        self.insert_into_pool(performance, config)

    def get_pool(self) -> List[Tuple[Performance, Prodict]]:
        with self.lock:
            return super().get_pool()

    def _find_clients(self, port, ip_folder: Union[Path, str]):
        """Parses the client directory to find all clients."""
        self.me = self.Client(socket.gethostbyname(socket.gethostname()), port, self.rank)
        if self.verbose > 1:
            print(f"My client: {self.me}")

        # We'll just read the ip files and get the task ids from there.
        # TODO: really should use os.environ?
        for f in Path(ip_folder).iterdir():
            rank, ip = f.read_text().rstrip("\n").split(" ")
            if rank == str(self.rank):
                # Ignore our rank
                assert self.me.ip == ip
                continue
            client = self.Client(ip, port, int(rank))
            self.clients.append(client)
            if self.verbose > 1:
                print(f"Found client: {client}")

    def _start_server(self, host: str, port: int):
        server = self.Server(host, port)
        server.start(len(self.clients))

        while True:
            conn, addr = server.accept()
            if self.verbose > 0:
                print(f"{self.rank}: Got connection from {addr}...")

            with self.lock:
                if self.stop:
                    print("Received stop signal. Closing socket...")
                    break

            data = server.receive(conn)
            performance, config = pickle.loads(data)

            if config.rank not in [client.rank for client in self.clients]:
                print(f"Received connection from unknown client ({config.rank}: {addr}). Adding to list.")
                assert hasattr(config, "rank")
                assert all([config.rank != client.rank for client in self.clients])
                self.clients.append(self.Client(addr, server.socket.getsockname()[1], config.rank))

            self.insert_into_pool(performance, config)

    def close(self):
        if self.verbose > 1:
            print(f"{self.task_id}: Closing socket...")

        with self.lock:
            self.stop = True

        self.me.connect()

        if self.verbose > 2:
            print(f"{self.task_id}: Closed socket.")

        if self.thread is not None:
            if self.verbose > 2:
                print(f"{self.task_id}: Joining thread...")
            self.thread.join()
            if self.verbose > 2:
                print(f"{self.task_id}: Joined thread.")
        if self.verbose > 1:
            print(f"{self.task_id}: Closed socket.")


class SharedFileAgentPool(AgentPool):
    """This AgentPool implements the storage mechanism as a shared file."""

    def __init__(self, config: Prodict, rank: int, *, verbose: int = 0):
        super().__init__(config, rank, verbose=verbose)

        self.filename = (
            Path(self.config.env_config.logdir)
            / self.config.env_config.exp_name
            / self.config.evo_config.agent_pool_shared_filename
        )
        self.lock_file = self.filename.with_suffix(".lock")
        self.lock_file.touch()
        try:
            fcntl.flock(open(self.lock_file, "w"), fcntl.LOCK_EX)
            fcntl.flock(open(self.lock_file, "w"), fcntl.LOCK_UN)
        except OSError:
            print("ERROR: This OS doesn't support fcntl. Pick another agent pool type.")
            exit(1)

        with self:
            self.filename.touch()

    def write_to_pool(self, performance: Performance, config: Prodict):
        with self:
            if self.filename.stat().st_size > 0:
                with open(self.filename, "rb") as f:
                    self.pool = pickle.load(f)

            self.insert_into_pool(performance, config)

            with open(self.filename, "wb") as f:
                pickle.dump(self.pool, f)

    def get_pool(self) -> List[Tuple[Performance, Prodict]]:
        with self:
            with open(self.filename, "rb") as f:
                self.pool = pickle.load(f)
        return super().get_pool()

    def acquire_lock(self):
        # fcntl NOT SUPPORTED ON SUPERCLOUD
        fcntl.flock(open(self.lock_file, "w"), fcntl.LOCK_EX)

    def release_lock(self):
        # fcntl NOT SUPPORTED ON SUPERCLOUD
        fcntl.flock(open(self.lock_file, "w"), fcntl.LOCK_UN)

    def __enter__(self):
        self.acquire_lock()
        return self

    def __exit__(self, *args):
        self.release_lock()


class AgentPoolCallback(BaseCallback):
    parent: EvalCallback

    def __init__(self, agent: Agent, agent_pool: AgentPool, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.agent = agent
        self.agent_pool = agent_pool

    def _on_step(self) -> bool:
        if self.parent is not None:
            self.agent_pool.write_to_pool(
                self.parent.best_mean_reward, self.agent.config
            )
