import click
from utils import predict, save_evolution, read_init, plot_evolution

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output')
@click.option('--visualize', is_flag=True,
             help='Add visualisation of evolution to output')
def cli(input, output, visualize):
    Ed = read_init(input)
    evolution = predict(Ed)
    save_evolution(evolution, output)

    if visualize:
        plot_evolution(evolution, output)


