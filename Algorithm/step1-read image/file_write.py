
def write_ndarray():
    """
    넘파이 배열을 파일에 쓰기
    """
    import numpy as np

    List = np.arange(start=5, stop=23, step=2, dtype=int)

    file_path = "./note/example.txt"

    np.savetxt(fname=file_path, X=List, fmt='%d')

def write_array():
    """
    배열을 파일에 쓰기
    """
    List = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]

    file_path = './note/example2.txt'

    with open(file=file_path, mode='w') as file:
        for item in List:
            file.write(str(item) + '\n')

    file.close()

def write_list(list, file_path):
    
    with open(file=file_path, mode='w') as file:
        for item in list:
            file.write(str(item) + '\n')