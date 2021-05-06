from typing import NamedTuple, Tuple


class TextFormat(NamedTuple):
    """Propiedades del formato de un texto."""
    font: int
    color: Tuple[int, int, int]
    linetype: int
    thickness: int
    font_scale: float
    bottom_left_origin: bool = False
