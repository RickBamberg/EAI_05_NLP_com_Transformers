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

    pad = curses.newpad(100, 100)
    stdscr.refresh()
    
    for i in range(100):
        for j in range(26):
            char = chr(67 + j )
            pad.addstr(char, GREEN_AND_GREY)

    pad.refresh(5, 5, 5, 5, 10, 25)
    stdscr.getch()
	
wrapper(main)