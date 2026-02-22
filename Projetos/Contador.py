import curses
from curses import wrapper
import time

def main(stdscr):
    curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_YELLOW)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_WHITE)
    BLUE_AND_YELLOW = curses.color_pair(1)
    GREEN_AND_GREY = curses.color_pair(2)
    ORANGE_AND_WHITE = curses.color_pair(3)

    for i in range(100):
        stdscr.clear()
        color = BLUE_AND_YELLOW

        if i % 2 == 0:
            color = GREEN_AND_GREY

        stdscr.addstr(10, 10, f'Count: {i}', color)
        stdscr.refresh()
        time.sleep(0.1)

    stdscr.getch()
	
wrapper(main)