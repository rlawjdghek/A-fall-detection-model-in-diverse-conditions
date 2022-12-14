import numpy as np

def levenshtein(s1, s2):
    """
    두 단어의 형태소 거리
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
def greedy_algorithm(mat):
    result = []
    for i,  r in enumerate(mat):
        result.append((i, r.argmin()))
    return result
        
def matching_greedy(ocr_results, labels):
    """
    labels에 있는 것만으로 ocr_results중 가장 비슷한 문자열과 비교해서 맞춤. 중복 허용 o
    """
    n_label = len(labels)
    n_pred = len(ocr_results)
    d_mat = np.zeros((n_label, n_pred))
    for i, l in enumerate(labels):
        for j, r in enumerate(ocr_results):
            _, text, prob = r
            d = levenshtein(l, text)
            d_mat[i][j] = d
    min_pos = greedy_algorithm(d_mat)
    new_ocr_results = []
    for label_idx, ocr_idx in min_pos:
        bbox, _, prob = ocr_results[ocr_idx]
        new_r = (bbox, labels[label_idx], prob)
        new_ocr_results.append(new_r)
    return new_ocr_results, ocr_results
    
ocr_results = [([[1448, 154], [1920, 154], [1920, 306], [1448, 306]], '증명여권비자', 0.48629955347520376), ([[190, 401], [360, 401], [360, 474], [190, 474]], '김희반찬포유', 0.8092117154957137), ([[400, 399], [527, 399], [527, 466], [400, 466]], '만수무병', 0.4853135347366333), ([[612, 370], [824, 370], [824, 458], [612, 458]], '글라스박스안경', 0.6860029208186303), ([[891, 333], [1230, 333], [1230, 457], [891, 457]], '행복누리약국', 0.6952205827946805), ([[1422, 304], [1750, 304], [1750, 430], [1422, 430]], '신한은행', 0.9573546648025513), ([[1028.3909097485723, 85.34636606971733], [1242.8120690194048, 57.27269214063176], [1247.6090902514277, 141.65363393028267], [1033.1879309805952, 169.72730785936824]], '위시티사', 0.9836810827255249), ([[362.6400127199618, 165.82402035193894], [758.1055912245934, 105.4540299657571], [767.3599872800381, 202.17597964806106], [371.8944087754066, 263.5459700342429]], '피부버디두피왁싱웨딩산전산후부', 0.196657300690595)]
labels = ['글라스박스안경', '신한은행', '행복온누리약국', '만수무병', '김숙희반찬포유']

new_ocr_results, ocr_results = matching_greedy(ocr_results, labels)
print(new_ocr_results)

