
def cross_validate(task, partitions):
    def inner(*args, **kwargs):
        for train, validate in partitions:
            yield task(train, validate, *args, **kwargs)
    return inner

def leave_one_cell_line_out(datas):
    for data in datas:
        regions = data.regions
        TF = os.path.basename(regions.path).split('.')[0]
        cell_lines = sorted(list(regions.cell_lines()))
        for cell_line in cell_lines:
            I = regions.regions[:,0] == cell_line
            train = data.copy()
            train.regions = regions[~I]
            validate = data.copy()
            validate.regions = regions[I]
            df = pd.DataFrame()
            df['TF'] = [TF]
            df['Heldout'] = [cell_line]
            yield df, train, validate






