import click
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import zarr


@click.command()
@click.option("--dataset", default="data")
@click.option("--index", default=0)
@click.option("--save", is_flag=True)
def main(dataset, index, save):
    root = zarr.group(dataset)

    ends = root.get_array("episode_ends")[:]
    ends = np.asarray(ends, dtype=np.uint64)

    starts = np.concatenate([np.zeros(1, dtype=np.uint64), ends[:-1]])

    rr.init(f"{index}", spawn=not save)

    action_group = root.get_group("action")

    action = {}

    for name in action_group.array_keys():
        z = action_group.get_array(name)[starts[index] : ends[index]]
        action[name] = np.asarray(z, dtype=np.float64)

    obs_group = root.get_group("obs")

    obs = {}

    for name in obs_group.array_keys():
        z = obs_group.get_array(name)[starts[index] : ends[index]]
        if "img" in name:
            obs[name] = np.asarray(z, dtype=np.uint8)
        else:
            obs[name] = np.asarray(z, dtype=np.float64)

    times = obs["timestamp"][starts[index] : ends[index]]
    times = np.asarray(times, np.float64)
    times -= times[0]

    for i in range(7):
        qpos = obs["qpos"][:]
        qvel = obs["qvel"][:]

        rr.send_columns(
            f"obs/qpos/{i}",
            indexes=[rr.TimeColumn("time", duration=times)],
            columns=rr.Scalars.columns(scalars=qpos[:, i]),
        )

        rr.send_columns(
            f"obs/qvel/{i}",
            indexes=[rr.TimeColumn("time", duration=times)],
            columns=rr.Scalars.columns(scalars=qvel[:, i]),
        )

    img = obs["wrist_img"].view(np.uint8)

    format = rr.components.ImageFormat(
        width=img.shape[2],
        height=img.shape[1],
        color_model="RGB",
        channel_datatype="U8",
    )
    rr.log("obs/wrist_img", rr.Image.from_fields(format=format), static=True)

    rr.send_columns(
        "obs/wrist_img",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Image.columns(buffer=img.reshape(len(times), -1)),
    )

    img = obs["front_img"].view(np.uint8)

    format = rr.components.ImageFormat(
        width=img.shape[2],
        height=img.shape[1],
        color_model="RGB",
        channel_datatype="U8",
    )
    rr.log("obs/front_img", rr.Image.from_fields(format=format), static=True)

    rr.send_columns(
        "obs/front_img",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=rr.Image.columns(buffer=img.reshape(len(times), -1)),
    )

    for i in range(7):
        qpos = action["qpos"][:]

        rr.send_columns(
            f"action/qpos/{i}",
            indexes=[rr.TimeColumn("time", duration=times)],
            columns=rr.Scalars.columns(scalars=qpos[:, i]),
        )

    blueprint = rrb.Grid(
        rrb.TimeSeriesView(
            name="Joint Positions",
            origin="obs/qpos",
        ),
        rrb.TimeSeriesView(name="Joint Velocities", origin="obs/qvel"),
        rrb.TimeSeriesView(name="Action Joint Positions", origin="action/qpos"),
        rrb.Spatial2DView(name="Wrist Camera", origin="obs/wrist_img"),
        rrb.Spatial2DView(name="Front Camera", origin="obs/front_img"),
    )

    rr.send_blueprint(blueprint)

    if save:
        rr.save(f"{index}.rrd")


if __name__ == "__main__":
    main()
