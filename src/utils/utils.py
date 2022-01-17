def format_time(t):
    ''' 
    Format seconds to days, hours, minutes, and seconds.
    -> Output format example: "01d-09h-24m-54s"
    
    Args:
        t (float, int): Time in seconds.
    '''
    assert isinstance(t, (float, int))

    h, r = divmod(t,3600)
    d, h = divmod(h, 24)
    m, r = divmod(r, 60)
    s, r = divmod(r, 1)

    values = [d, h, m, s]
    symbols = ['d', 'h', 'm', 's']
    for i, val in enumerate(values):
        if val > 0:
            symbols[i] = ''.join([f'{int(val):02d}', symbols[i]])
        else:
            symbols[i] = ''
    return '-'.join(s for s in symbols if s) if any(symbols) else '<1s'