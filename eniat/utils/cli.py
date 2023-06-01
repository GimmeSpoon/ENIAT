from shutil import get_terminal_size

def introduce() -> None:
    '''Font Design by myflix, and modified by patorjk'''
    w, h = get_terminal_size()
    lw = 49
    lc = max((w-lw)//2,0)

    print('\n' + '*' * w + '\n')
    print(' ' * lc + "   ('-.       .-') _           ('-.     .-') _")
    print(' ' * lc + " _(  OO)     ( OO ) )         ( OO ).-.(  OO) )")
    print(' ' * lc + "(,------.,--./ ,--,' ,-.-')   / . --. //     '._")
    print(' ' * lc + " |  .---'|   \ |  |\ |  |OO)  | \-.  \ |'--...__)")
    print(' ' * lc + " |  |    |    \|  | )|  |  \.-'-'  |  |'--.  .--'")
    print(' ' * lc + "(|  '--. |  .     |/ |  |(_/ \| |_.'  |   |  |")
    print(' ' * lc + " |  .--' |  |\    | ,|  |_.'  |  .-.  |   |  |")
    print(' ' * lc + " |  `---.|  | \   |(_|  |     |  | |  |   |  |")
    print(' ' * lc + " `------'`--'  `--'  `--'     `--' `--'   `--'")

    print(' ' * max((w-9)//2,0) + 'E N I A T')
    print(' ' * max((w-10)//2,0) + 'Ver 0.1.0')
    print(' ' * max((w-45)//2,0) + 'Eccentric Nimble Immaculate Advanced Template\n')
    print(' ' * max((w-47)//2,0) + 'Font designed by myflix and modified by patorjk\n')
    print('*' * w + '\n')