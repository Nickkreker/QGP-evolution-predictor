import click
import time
from utils import predict, save_evolution, read_init, plot_evolution

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output')
@click.option('--visualize', is_flag=True,
             help='Add visualisation of evolution to output')
@click.option('--verbose', '-v', is_flag=True,
             help='Print additional runtime information')
def cli(input, output, visualize, verbose):
    start_time = time.perf_counter()
    if verbose:
        print('Reading input')
    
    Ed = read_init(input)

    if verbose:
        print('Calculating evolution')
    evolution = predict(Ed)

    if verbose:
        print('Saving evolution to file')
    save_evolution(evolution, output)

    if visualize:
        plot_evolution(evolution, output)

    elapsed_time = time.perf_counter() - start_time
    if verbose:
        print(f'Runtime={elapsed_time}s')


