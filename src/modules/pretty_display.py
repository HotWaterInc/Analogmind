import sys

total_number_per_line = None
pretty_display_name = None
current_line = None
last_percentage = None


def set_pretty_display(total_number_per_line_value, name):
    global total_number_per_line \
        , pretty_display_name, current_line, last_percentage

    total_number_per_line = total_number_per_line_value
    pretty_display_name = name
    current_line = False
    last_percentage = 0


def _display_bars(current_percentage, prev_percentage):
    current_percentage -= prev_percentage
    if current_percentage == 0:
        return

    for i in range(current_percentage):
        print("#", end="")
        sys.stdout.flush()


def pretty_display(current_value):
    global total_number_per_line, pretty_display_name, current_line, last_percentage
    percentage = int(current_value / total_number_per_line * 100)
    # print("current_value", current_value, "percentage", percentage, "last_percentage", last_percentage)
    if percentage == last_percentage:
        return

    _display_bars(percentage, last_percentage)
    last_percentage = percentage


def pretty_display_reset():
    global current_line, last_percentage
    current_line = False
    last_percentage = 0
    print("")


def pretty_display_start(current_line_value):
    global current_line
    print(f"{pretty_display_name}: {current_line_value} loading:", end="")
    current_line = True
