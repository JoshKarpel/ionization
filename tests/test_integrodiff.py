import ionization.integrodiff as ide


def test_hydrogen_kernel_LEN_factory_returns_singleton():
    first = ide._hydrogen_kernel_LEN_factory()
    second = ide._hydrogen_kernel_LEN_factory()

    print(first, second)

    assert first is second
