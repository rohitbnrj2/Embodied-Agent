#!/bin/bash

max() {
    (( $1 > $2 )) && echo $1 || echo $2
}

min() {
    (( $1 < $2 )) && echo $1 || echo $2
}

# Function to remove an argument and its value from the arguments array; thanks chatgpt
remove_arg_and_value() {
    local arg_to_remove=$1
    local new_args=()
    local skip_next=false
    for arg in "$@"; do
        if $skip_next; then
            skip_next=false
            continue
        fi
        if [[ $arg == $arg_to_remove ]]; then
            skip_next=true
            continue
        fi
        new_args+=("$arg")
    done
    echo "${new_args[@]}"
}

configs="domain_randomization/domain_randomization domain_randomization/domain_randomization_random_textures"
experiments="point_2eyes_10x10 point_16eyes_10x10 point_2eyes_2x2 point_16eyes_2x2"
n_temporal_obs="1 5 10"
dry_run=false

# Iterate over arguments
# Check/remove --experiments argument
for arg in "$@"; do
    if [[ $arg == "--experiments="* ]]; then
        experiments="${arg#*=}"
        set -- $(remove_arg_and_value "--experiments=$experiments" "$@")
    elif [[ $arg == "--configs="* ]]; then
        configs="${arg#*=}"
        set -- $(remove_arg_and_value "--configs=$configs" "$@")
    elif [[ $arg == "--n-temporal-obs="* ]]; then
        n_temporal_obs="${arg#*=}"
        set -- $(remove_arg_and_value "--n-temporal-obs=$n_temporal_obs" "$@")
    elif [[ $arg == "--dry-run" ]]; then
        dry_run=true
        set -- $(remove_arg_and_value "--dry-run" "$@")
    fi
    let i++
done

for config in $configs; do
    for experiment in $experiments; do
        include='${extend:${include:configs_mujoco/experiments/'${experiment}'.yaml}}'
        for n_temporal_obs in $n_temporal_obs; do
            exp_name="${config}_2_${experiment}_n${n_temporal_obs}"

            # If calc num pixels by taking num eyes and multiplying by num pixels (i.e. 10x10 = 100 pixels)
            num_eyes=$(echo "$experiment" | cut -d'_' -f2 | grep -o -E '[0-9]+')
            num_width=$(echo "$experiment" | cut -d'_' -f3 | cut -d'x' -f1)
            num_height=$(echo "$experiment" | cut -d'_' -f3 | cut -d'x' -f2)
            num_pixels=$(($num_eyes * $num_width * $num_height))
            n_envs=$(min 10 $(( 3200 / $num_pixels )))

            echo "Running experiment ${experiment} with ${config} and ${n_temporal_obs} temporal observation(s): ${exp_name}"

            # Only run if not dry run. Check if NOT dry run
            if ! $dry_run; then
                cmd="MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
                    configs_mujoco/experiments/${config}.yaml --train \
                    -o extend.experiment='${include}' \
                    -o training_config.total_timesteps=5_000_000 \
                    -ao n_temporal_obs=${n_temporal_obs} \
                    -o training_config.exp_name="${exp_name}" \
                    -o training_config.n_envs=${n_envs}
                    $@"
                echo "Running: " $cmd
                if ! eval $cmd; then
                    echo "Error running command. Stopping..."
                    exit 1
                fi
            fi 
        done
    done
done