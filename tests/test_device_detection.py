import torch


def test_device_choice():
    """Ensure that we can inspect torch to find a device fallback."""
    # if no CUDA available, this should be False
    has_cuda = torch.cuda.is_available()
    assert isinstance(has_cuda, bool)
    # assert that device can be determined
    if has_cuda:
        assert torch.cuda.device_count() > 0
    else:
        assert torch.cuda.device_count() == 0
