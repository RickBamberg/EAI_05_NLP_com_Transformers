import curses
from curses import wrapper

def main(stdscr):
    curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_YELLOW)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_WHITE)
    BLUE_AND_YELLOW = curses.color_pair(1)
    GREEN_AND_GREY = curses.color_pair(2)
    ORANGE_AND_WHITE = curses.color_pair(3)

    stdscr.clear()
    stdscr.addstr(10, 10, "Rick", BLUE_AND_YELLOW | curses.A_UNDERLINE)
    stdscr.addstr(15, 15, "Rick", GREEN_AND_GREY)
    stdscr.addstr(20, 20, "Rick", ORANGE_AND_WHITE)
    stdscr.refresh()
    stdscr.getch()
	
wrapper(main)