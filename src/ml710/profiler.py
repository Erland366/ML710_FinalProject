import os
import time
import datetime
import torch
from torch.profiler import tensorboard_trace_handler

def trace_handler(
    prof,
    output_dir: str,
    metric: str = "self_cuda_time_total",
    row_limit: int = 25,
    run_name: str = "default",
):
    """
    Handles export of artifacts from ``torch.profiler.profile``.

    The following artifacts are exported:
    - chrome / tensorboard trace - viewable through tensorboard or perfetto.dev / chrome::/tracing
    - trace event table
    - memory timeline and snapshot.pickle if ``profile_memory``
    - stacks if ``with_stack`` (note that ``profile_memory`` requires ``with_stack`` to be ``True``),
    viewable as a flamegraph see (https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile.export_stacks).

    Notes:
    - Each profiling cycle is exported as a sub-directory in output_dir
        - E.g., profiling in 5-step cycle (wait=2, warmup=2, active=1, repeat=0) will result in
        sub-directories iteration_5, iteration_10, etc.
    - If profiling in a distributed setting, each artifact will be prefixed with rank.
    - Memory timeline is only exported for rank 0 (error if exporting from multiple ranks on single node)

    See profiler documentation (https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile) for more details

    Args:
        prof (torch.profiler.profile): instance of torch profiler to use
        output_dir (str):  directory to store artifacts
        metric (str): metric to order trace event table by, see ``torch.profiler.profile.key_averages().table`` for
        row_limit (int): number of rows to display in trace event table

    """

    world_size, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    curr_trace_dir_name = "iteration_" + str(prof.step_num)
    curr_trace_dir = os.path.join(output_dir, run_name, curr_trace_dir_name)
    if not os.path.exists(curr_trace_dir):
        os.makedirs(curr_trace_dir, exist_ok=True)

    # Export chrome / tensorboard trace
    if rank == 0:
        print(f"Dumping traces at step {prof.step_num}")
        # log.info(f"Dumping traces at step {prof.step_num}")
    begin = time.monotonic()

    # Use tensorboard trace handler rather than directly exporting chrome traces since
    # tensorboard doesn't seem to be able to parse traces with prof.export_chrome_trace

    now = datetime.datetime.now()

    exporter = tensorboard_trace_handler(
        curr_trace_dir,
        worker_name=f"r0-{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}",
        use_gzip=True,
    )
    exporter(prof)

    if rank == 0:
        # log.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")
        print(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")

    # Memory timeline sometimes fails to export
    if prof.profile_memory:
        if rank == 0:
            try:
                prof.export_memory_timeline(
                    f"{curr_trace_dir}/{run_name}_rank{rank}_memory-timeline.html"
                )
            except Exception as e:
                # log.warn(f" Failed to export memory timeline: {e}")
                print(f"Saving profiling results to {curr_trace_dir}")

            torch.cuda.memory._dump_snapshot(
                f"{curr_trace_dir}/{run_name}_rank{rank}_memory_snapshot.pickle"
            )

    # Dump stack traces
    if prof.with_stack:
        prof.export_stacks(f"{curr_trace_dir}/{run_name}_rank{rank}_stacks.txt", metric=metric)

    # Export event averages
    key_avgs = prof.key_averages(
        group_by_input_shape=prof.record_shapes, group_by_stack_n=5
    ).table(sort_by=metric, row_limit=row_limit)
    with open(f"{curr_trace_dir}/{run_name}_rank{rank}_key_averages.txt", "w") as f:
        print(key_avgs, file=f)
    if rank == 0:
        # log.info(f"Saving profiling results to {curr_trace_dir}")
        print(f"Saving profiling results to {curr_trace_dir}")

    # TODO: Is this necessary?
    # see https://github.com/pytorch/torchtitan/blob/3050098dcee4901d88c712f9e8e9703d1735a29b/torchtitan/profiling.py#L48
    torch.distributed.barrier()