from shutil import get_terminal_size

def introduce() -> None:
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
    print('*' * w + '\n')

def help() -> None:
    w, h = get_terminal_size()
    lc = 10
    print('           ENIAT - Eccentric, Nimble, Immaculate, Advanced, Template')
    print('           Template for various ML libraries suchas pytorch, scikit-learn')
    print('           Version 0.1.0 (2023-04-12)')
    print('           maintained by GimmeSpoon@github\n')
    print('           This app is powered by Hydra, and this message does not include basic Hydra information.')
    print('           If you are not familiar with Hydra or ML libraries you intend to use,')
    print('           please refer to the official documentation.\n')
    print('           -- Arguments --')
    print('           []: list, ():options\n')
    print("           do=('init','help') :")
    print('               init: skip default tasks and make directories for config.')
    print('               help: skip default tasks and print out the help message\n')
    print('           config=[PATH_TO_CONFIG_FILE] : paths of config files to merge\n')
    print("           output_dir=PATH_TO_OUTPUT_DIR : path to the output directory (default: './result')")
    print("           ")