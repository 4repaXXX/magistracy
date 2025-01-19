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

def find_min_n_for_H1(sequence):
    """
    Ищем минимальное n для H1-нарезки:
    (подпоследовательность длины n, следующий отсчёт)
    должны быть уникальными.
    Формально k от 0 до M−(n+1)
    Пары ((x_k​,…,x_{k+n−1}​),x_{k+n}​)
    """
    length = len(sequence)
    for n in range(1, length):
        seen = set()
        unique = True
        for k in range(length - n):
            window = tuple(sequence[k:k+n])   
            next_val = sequence[k+n]
            pair = (window, next_val)
            if pair in seen:
                unique = False
                break
            else:
                seen.add(pair)
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

    result_n = find_min_n_for_H1(seq)
    
    print(f"Минимальная длина неповторяющихся подпоследовательностей = {result_n}")
