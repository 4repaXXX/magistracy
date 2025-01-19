import sys

def read_sequence_from_file(filename):
    sequence = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                stripped = line.strip()
                if stripped:
                    try:
                        number = float(stripped)
                        sequence.append(number)
                    except ValueError:
                        print(f"Предупреждение: не удалось преобразовать строку '{stripped}' в число.")
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден.")
        sys.exit(1)
        
    print(f"Последовательность {sequence}")
    return sequence


def find_min_unique_length(sequence, round_digits=None):
    """
    Определяет минимальное n, при котором все подпоследовательности длины n
    в 'sequence' являются уникальными (не повторяются).
    
    Параметры:
    ----------
    sequence - список с данным, вещественные числа
    
    Возвращает:
    int : искомая длина уникальных подпоследовательностей
    """
    length = len(sequence)
    
    if round_digits is not None:
        sequence = [round(x, round_digits) for x in sequence]

    for n in range(1, length + 1):
        seen = set()
        unique = True
        for start in range(0, length - n + 1):
            subseq = tuple(sequence[start:start+n])
            if subseq in seen:
                unique = False
                break
            else:
                seen.add(subseq)
        if unique:
            return n
    return length

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python *.py <путь_к_файлу>")
        sys.exit(1)

    filename = sys.argv[1]
    seq = read_sequence_from_file(filename)
    round_digits = 5

    result_n = find_min_unique_length(seq, round_digits)
    
    print(f"Минимальная длина неповторяющихся подпоследовательностей = {result_n}")
