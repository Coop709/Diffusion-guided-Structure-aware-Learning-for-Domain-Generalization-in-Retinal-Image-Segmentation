"""Pair source batches with a cycling stream of style images."""


def paired_batches(source_loader, style_loader):
    style_iterator = iter(style_loader)
    for source in source_loader:
        try:
            style = next(style_iterator)
        except StopIteration:
            style_iterator = iter(style_loader)
            try:
                style = next(style_iterator)
            except StopIteration as error:
                raise ValueError("The style-image loader is empty") from error
        yield source, style
