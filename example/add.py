import fire


def add_value(a: int, b: int) -> int:
    """二つの和を計算します

    usage: python add.py 1 2
    docstring は https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring などで生成すると捗ります。

    Args:
        a (int): 左辺値
        b (int): 右辺値

    Returns:
        int: 計算結果
    """
    return a + b


if __name__ == "__main__":
    fire.Fire(add_value)
