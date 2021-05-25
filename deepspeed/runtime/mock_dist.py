def is_available():
    return True


def is_initialized():
    return True


class ReduceOp:
    SUM = 0
    MAX = 1


class group:
    WORLD = 1


def _get_global_rank(group, group_rank):
    return 0


def get_rank(group=None):
    return 0


def get_world_size(group=None):
    return 1


class async_mock():
    @staticmethod
    def wait():
        return


def reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
    if async_op:
        return async_mock()


def barrier(group=group.WORLD, async_op=False, device_ids=None):
    if async_op:
        return async_mock()


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    if async_op:
        return async_mock()


def broadcast(tensor, src, group=None, async_op=False):
    if async_op:
        return async_mock()


def all_gather(tensor_list, tensor, group=None, async_op=False):
    if async_op:
        return async_mock()


def new_group(ranks=None, timeout=0, backend=None):
    assert len(ranks) == 1, "only supports 1 rank"
    return None
