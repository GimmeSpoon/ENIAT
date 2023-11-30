from types import ModuleType
from rich._log_render import FormatTimeCallable
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from rich.console import Console, ConsoleOptions, RenderResult, Group
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.text import Text
from rich.logging import RichHandler
from rich.highlighter import RegexHighlighter, Highlighter
from omegaconf import DictConfig
from typing import Iterable, List, Literal, Optional, Union
from contextlib import nullcontext
from dataclasses import dataclass, field
from logging import Formatter, LogRecord, StreamHandler
from logging.handlers import RotatingFileHandler
import logging
from math import ceil
from re import sub

SPINNER_STATUS = "dots"
SPINNER_TRAIN = "monkey"
SPINNER_EVAL = "moon"

REFRESH_FREQ = 12.5
PANEL_WIDTH = 100

_FILE_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
_RICH_FORMAT = "%(message)s"
_STRM_FORMAT = lambda x: f"[cyan dim]%(asctime)s[/] [{x}]%(levelname)s[/] %(message)s"
_DATE_FORMAT = "%y-%m-%d %H:%M:%S"


class LogHandler(logging.Handler):
    colors = {
        10: "green",  # DEBUG
        20: "blue",  # INFO
        30: "yellow",  # WARNING
        40: "red",  # ERROR
        50: "bold red",  # CRITICAL
    }

    def __init__(self, level=0) -> None:
        super().__init__(level)

    def highlight(self, msg: str):
        return sub(
            "\bEvaluation Result\b",
            "\b[bold red]Evaluation Result[/]\b",
            sub("\bTraining State\b", "\b[bold green]Training State[/]\b", msg),
        )

    def emit(self, record: LogRecord) -> None:
        self.setFormatter(
            Formatter(_STRM_FORMAT(self.colors[record.levelno]), _DATE_FORMAT)
        )
        logview.flush(self.highlight(self.format(record)) + "\n")


class LogFileHandler(RotatingFileHandler):
    def __init__(
        self,
        filename,
        mode: str = "a",
        maxBytes: int = 0,
        backupCount: int = 0,
        encoding: str | None = None,
        delay: bool = False,
        errors: str | None = None,
    ) -> None:
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay, errors)
        self.setFormatter(Formatter(_FILE_FORMAT, _DATE_FORMAT))


class PreLogHandler(RichHandler):
    def __init__(
        self,
        level: int | str = logging.NOTSET,
        console: Console | None = None,
        *,
        show_time: bool = True,
        omit_repeated_times: bool = True,
        show_level: bool = True,
        show_path: bool = True,
        enable_link_path: bool = True,
        highlighter: Highlighter | None = None,
        markup: bool = False,
        rich_tracebacks: bool = False,
        tracebacks_width: int | None = None,
        tracebacks_extra_lines: int = 3,
        tracebacks_theme: str | None = None,
        tracebacks_word_wrap: bool = True,
        tracebacks_show_locals: bool = False,
        tracebacks_suppress: Iterable[str | ModuleType] = ...,
        locals_max_length: int = 10,
        locals_max_string: int = 80,
        log_time_format: str | FormatTimeCallable = "[%x %X]",
        keywords: List[str] | None = None,
    ) -> None:
        super().__init__(
            level,
            console,
            show_time=show_time,
            omit_repeated_times=omit_repeated_times,
            show_level=show_level,
            show_path=show_path,
            enable_link_path=enable_link_path,
            highlighter=highlighter,
            markup=markup,
            rich_tracebacks=rich_tracebacks,
            tracebacks_width=tracebacks_width,
            tracebacks_extra_lines=tracebacks_extra_lines,
            tracebacks_theme=tracebacks_theme,
            tracebacks_word_wrap=tracebacks_word_wrap,
            tracebacks_show_locals=tracebacks_show_locals,
            tracebacks_suppress=tracebacks_suppress,
            locals_max_length=locals_max_length,
            locals_max_string=locals_max_string,
            log_time_format=log_time_format,
            keywords=keywords,
        )
        self.setFormatter(Formatter(_RICH_FORMAT, _DATE_FORMAT))


@dataclass
class LogView:
    lines: list[str] = field(default_factory=list)

    def __render(self, width: int, height: int):
        rendered = []
        wlmt, hlmt = max(width - 4, 1), max(height - 5, 1)
        for line in self.lines:
            num_line = ceil(len(line) / wlmt)
            if num_line <= 1:
                rendered.append(line)
            else:
                rendered += [line[wlmt * l : wlmt * (l + 1)] for l in range(num_line)]
        return rendered[-hlmt:]

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield "\n".join(self.__render(options.size.width, options.size.height))

    def flush(self, msg: str):
        self.lines += msg.strip().split("\n")
        live.refresh()

def progress_bar(spinner: str, **kwargs):
    return Progress(
        TextColumn("{task.description}"),
        SpinnerColumn(spinner),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        **kwargs,
        refresh_per_second=REFRESH_FREQ,
        expand=True,
    )


def init_display(console: Console = None, silent: bool = False):
    if console is None:
        console = Console()

    global status, progress1, progress2, ebar, sbar, vbar, g_console, live, main_layout, logview

    g_console = console
    status = console.status(
        "Initiating...", spinner=SPINNER_STATUS, refresh_per_second=REFRESH_FREQ
    )
    progress1 = progress_bar(SPINNER_TRAIN, console=console)
    progress2 = progress_bar(SPINNER_EVAL, console=console)

    logview = LogView()

    if silent:

        live = Live(Group(status, progress1, progress2))
        live.start()

    else:
        main_layout = Layout()
        main_layout.split_column(
            Layout(name="Log"),
            Layout(name="Progress", size=5),
        )

        main_layout["Progress"].update(
            Panel(Group(status, progress1, progress2), width=PANEL_WIDTH)
        )
        main_layout["Log"].update(logview)

        live = Live(main_layout, console=g_console, refresh_per_second=REFRESH_FREQ)
        live.start()

    ebar = progress1.add_task("Epoch", visible=False)
    sbar = progress1.add_task("Update Step", visible=False)
    vbar = progress2.add_task("Inference", visible=False)

    return console

#PROGRESS

def bar(
    task: Literal["train", "eval"],
    total_epochs=None,
    total_steps=None,
    start_epoch=0,
    start_step=0,
    msg=None,
):
    global ebar_restore

    ebar_restore = total_epochs is not None

    if task == "train":
        status.update("Training...") if msg is None else status.update(msg)
        if total_epochs is not None:
            progress1.update(ebar, total=total_epochs, completed=start_epoch, visible=True)
        progress1.update(sbar, total=total_steps, completed=start_step, visible=True)
    else:
        status.update("Evaluating...") if msg is None else status.update(msg)
        progress1.update(ebar, visible=False)
        progress1.update(sbar, visible=False)
        progress2.reset(vbar, total=total_steps, completed=start_step, visible=True)

    return nullcontext()


def advance(bar_type: Literal["epoch", "step", "eval"], steps: int = 1) -> None:
    if bar_type == "epoch":
        progress1.advance(ebar, steps)
    elif bar_type == "step":
        progress1.advance(sbar, steps)
    else:
        progress2.advance(vbar, steps)


def reset(bar_type: Literal["epoch", "step", "eval"], steps: int = 0) -> None:
    if bar_type == "epoch":
        progress1.reset(ebar, completed=steps)
    elif bar_type == "step":
        progress1.reset(sbar, completed=steps)
    else:
        progress2.reset(vbar, completed=steps)


def end(task: Literal["train", "eval"], restore: bool = False):
    if task == "train":
        status.update(f"Completed!", spinner="smiley")
        progress1.stop()
        progress2.stop()
    else:
        progress2.stop()
        if restore:
            progress2.update(vbar, visible=False)
            status.update("Training...")
            if ebar_restore:
                progress1.update(ebar, visible=True)
            progress1.update(sbar, visible=True)
        else:
            progress1.stop()
            status.update("Completed!", spinner="smiley")

def initiated() -> bool:
    global g_console
    return g_console is None
