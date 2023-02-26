import pandas as pd
import PTwithTimeTrend_AllStock as ptm
from johansen_class import Johansen
import time


def test_python():
    num = 40
    for num in range(0, 16):
        print('======================================')
        print("Nmm :", num)
        file = pd.read_csv(f"./check_data/check_data_{num}.csv", index_col=0)
        file = file.to_numpy()
        start = time.process_time()
        table = ptm.refactor_formation_table(file)
        output = pd.read_csv(
            f"./output_data/output_data_{num}.csv", index_col=0)
        print(table)
        print(output)
        end = time.process_time()
        print("執行時間：%f 秒" % (end - start))
    print("WWWWWWWWWWW")
    


if __name__ == "__main__":
    test_python()
