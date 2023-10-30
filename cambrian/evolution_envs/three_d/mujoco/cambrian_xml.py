from typing import List, Tuple
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


class MjCambrianXML:
    """Helper class for manipulating mujoco xml files. Provides some helper methods for
    that wrap the `xml` library.

    Args:
        base_xml_path (Path | str): The path to the base xml file to load.
    """

    def __init__(self, base_xml_path: Path | str):
        self._base_xml_path = Path(base_xml_path)

        self.load(self._base_xml_path)

    def load(self, path: Path | str):
        """Load the xml from a file."""
        self._tree = ET.parse(path)
        self._root = self._tree.getroot()

    def write(self, path: Path | str):
        """Write the xml to a file."""
        self._tree.write(path)

    @staticmethod
    def make_empty() -> "MjCambrianXML":
        """Loads an empty mujoco xml file. Only has the `mujoco` and worldbody
        `tags`."""
        return MjCambrianXML.from_string("<mujoco><worldbody></worldbody></mujoco>")

    @staticmethod
    def from_string(xml_string: str) -> "MjCambrianXML":
        """Loads the xml from a string."""
        with tempfile.NamedTemporaryFile("w") as f:
            f.write(xml_string)
            f.flush()
            return MjCambrianXML(f.name)

    def add(self, parent: ET.Element, tag: str, *args, **kwargs) -> ET.Element:
        """Add an element to the xml tree.

        Args:
            parent (ET.Element): The parent element to add the new element to.
            tag (str): The tag of the new element.
            *args: The arguments to pass to the `ET.SubElement` call.
            **kwargs: The keyword arguments to pass to the `ET.SubElement` call.
        """
        return ET.SubElement(parent, tag, *args, **kwargs)

    def remove(self, parent: ET.Element, element: ET.Element):
        """Remove an element from the xml tree.

        Args:
            parent (ET.Element): The parent element to remove the element from.
            element (ET.Element): The element to remove.
        """
        parent.remove(element)

    def find(
        self, tag: str, *, _all: bool = False, **kwargs
    ) -> List[ET.Element] | ET.Element | None:
        """Find an element by tag.

        If any additional keyword arguments are passed, they will be used to filter the
        elements, as in the element must have the attribute and it must be equal to the
        value. In this case, `ET.iterfind` will be used. It uses the predicates
        described here: https://docs.python.org/3/library/xml.etree.elementtree.html#supported-xpath-syntax.

        If no keyword arguments are passed, `ET.find` will be used.

        If not found, `None` will be returned.

        Args:
            tag (str): The tag of the element to find.
            _all (bool): If true, will use `ET.findall`, else `ET.find`.
            **kwargs: The keyword arguments to filter the elements by.

        Returns:
            List[ET.Element] | ET.Element | None: The element or `None` if not found. If
                `_all` is true, a list of elements will be returned.
        """
        kwargs_str = "".join([f"[@{key}='{value}']" for key, value in kwargs.items()])
        if _all:
            return self._root.findall(f"{tag}{kwargs_str}")
        else:
            return self._root.find(f"{tag}{kwargs_str}")

    def findall(self, tag: str, **kwargs) -> List[ET.Element]:
        """Alias for `find(tag, _all=True, **kwargs)`."""
        return self.find(tag, _all=True, **kwargs)

    def get_path(self, element: ET.Element) -> Tuple[List[ET.Element], str]:
        """Get the path of an element in the xml tree. Unfortunately, there is no
        built-in way to do this. We'll just iterate up the tree and build the path.

        Args:
            element (ET.Element): The element to get the path to.

        Returns:
            Tuple[List[ET.Element], str]: The list of elements in the path and the
            string representation of the path. The list of elements is ordered from root
            down to the passed element. NOTE: the root is _not_ included in the list.
        """

        path: List[str] = []
        elements: List[ET.Element] = []

        parent_map = {child: parent for parent in self._root.iter() for child in parent}
        while element is not None:
            if element == self._root:
                break

            path.insert(0, element.tag)
            elements.insert(0, element)
            element = parent_map.get(element)

        return elements, "/".join(path)

    def combine(self, root: ET.Element, other: ET.Element) -> ET.Element:
        """Combines two xml trees. Preferred to be called through the `+` or `+=`
        operators.

        Taken from here: https://stackoverflow.com/a/29896847/20125256
        """

        class hashabledict(dict):
            def __hash__(self):
                return hash(tuple(sorted(self.items())))

        # Create a mapping from tag name to element, as that's what we are fltering with
        mapping = {(el.tag, hashabledict(el.attrib)): el for el in root}
        for el in other:
            if len(el) == 0:
                # Not nested
                try:
                    # Update the text
                    mapping[(el.tag, hashabledict(el.attrib))].text = el.text
                except KeyError:
                    # An element with this name is not in the mapping
                    mapping[(el.tag, hashabledict(el.attrib))] = el
                    # Add it
                    root.append(el)
            else:
                try:
                    # Recursively process the element, and update it in the same way
                    self.combine(mapping[(el.tag, hashabledict(el.attrib))], el)
                except KeyError:
                    # Not in the mapping
                    mapping[(el.tag, hashabledict(el.attrib))] = el
                    # Just add it
                    root.append(el)
        return root

    @property
    def root(self) -> ET.Element:
        """The root element of the xml tree."""
        return self._root

    @property
    def base_dir(self) -> Path:
        """The directory of the base xml file."""
        return self._base_xml_path.parent

    def __add__(self, other: "MjCambrianXML"):
        assert isinstance(other, MjCambrianXML)
        self += other
        return self

    def __iadd__(self, other: "MjCambrianXML"):
        assert isinstance(other, MjCambrianXML)
        self._tree = ET.ElementTree(self.combine(self._root, other._root))
        return self

    def __str__(self):
        str = ET.tostring(self._root, encoding="unicode").replace("\n", "")
        return minidom.parseString(str).toprettyxml(indent=" ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MujocoXML Tester")

    parser.add_argument("xml_path1", type=str, help="The path to the first xml file.")
    parser.add_argument("xml_path2", type=str, help="The path to the second xml file.")

    args = parser.parse_args()

    xml1 = MjCambrianXML(args.xml_path1)
    xml2 = MjCambrianXML(args.xml_path2)
    xml = xml1 + xml2
    xml += xml2
    print(xml)
