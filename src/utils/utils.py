def format_time(time):
    ''' 
    Format seconds to days, hours, minutes, and seconds.
    -> Output format example: "01d-09h-24m-54s"
    
    Args:
        time (float, int): Time in seconds.
    '''
    assert isinstance(time, (float, int))

    hours, rest = divmod(time,3600)
    days, hours = divmod(hours, 24)
    minutes, rest = divmod(rest, 60)
    seconds, rest = divmod(rest, 1)

    values = [days, hours, minutes, seconds]
    symbols = ['d', 'h', 'm', 's']
    for i, val in enumerate(values):
        if val > 0:
            symbols[i] = ''.join([f'{int(val):02d}', symbols[i]])
        else:
            symbols[i] = ''
    return '-'.join(s for s in symbols if s) if any(symbols) else '<1s'

def print_epoch(epoch, time_elapsed, train_loss, validation_loss=None, improved=False):

    time_str = format_time(time_elapsed)
    symb = '*' if improved else ''

    train_print_string = f'Train Loss: {train_loss:>9.5f}'
    val_print_string = f'Validation Loss: {validation_loss:>9.5f}' if validation_loss is not None else ''
    
    console_printout = f'Epoch {epoch+1:>2} ({time_str}/it) -- ({train_print_string}{val_print_string}) {symb}'
    
    print(console_printout)
