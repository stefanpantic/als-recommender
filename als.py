import click

from modes.prepare_data import prepare_data
from modes.train import train
from modes.recommend import recommend
from modes.summary import summary

if __name__ == '__main__':
    cli = click.Group()
    cli.add_command(prepare_data)
    cli.add_command(train)
    cli.add_command(recommend)
    cli.add_command(summary)
    cli()
