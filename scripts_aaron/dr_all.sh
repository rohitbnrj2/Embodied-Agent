max() {
    (( $1 > $2 )) && echo $1 || echo $2
}

for config in dr dr_random_textures; do
    for experiment in point_2eyes_10x10 point_16eyes_10x10 point_2eyes_2x2 point_16eyes_2x2; do
        include='${extend:${include:configs_mujoco/experiments/'${experiment}'.yaml}}'
        for n_temporal_obs in 1 5 10; do
            # If calc num pixels by taking num eyes and multiplying by num pixels (i.e. 10x10 = 100 pixels)
            num_eyes=$(echo "$experiment" | cut -d'_' -f2 | grep -o -E '[0-9]+')
            num_width=$(echo "$experiment" | cut -d'_' -f3 | cut -d'x' -f1)
            num_height=$(echo "$experiment" | cut -d'_' -f3 | cut -d'x' -f2)
            num_pixels=$(($num_eyes * $num_width * $num_height))
            n_envs=$(max 40 $(( 3200 / $num_pixels )))

            echo "Running experiment ${experiment} with ${config} and ${n_temporal_obs} temporal observations"
            MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
                configs_mujoco/experiments/${config}.yaml --train \
                -o extend.experiment=${include} \
                -o training_config.total_timesteps=5_000_000 \
                -ao n_temporal_obs=${n_temporal_obs} \
                -o training_config.exp_name="dr_${experiment}_n${n_temporal_obs}" \
                -o training_config.n_envs=${n_envs}
        done
    done
done