import os
import time
from tqdm import tqdm
import numpy as np
import json
from .normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer
from .normalizer.english_abbreviations import english_spelling_normalizer
import Levenshtein
import editdistance
import concurrent.futures


# Function that generates ALL intermediate strings on the path from s1 to s2
def get_path_strings(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Fill the matrix
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Traceback (path restoration)
    # We collect operations to apply them to s1 later
    # We go from the end (n,m) to the beginning (0,0).
    # Operations: ('del', idx), ('ins', idx, char), ('sub', idx, char), ('match', idx)

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        cost = 0 if (i > 0 and j > 0 and s1[i - 1] == s2[j - 1]) else 1

        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + cost:
            # Diagonal (Match or Sub)
            if cost == 1:  # Substitution
                ops.append(('sub', i - 1, s2[j - 1]))
            else:
                pass  # Match - no changes needed
            i -= 1;
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Up (Deletion from s1)
            ops.append(('del', i - 1))
            i -= 1
        else:
            # Left (Insertion from s2)
            # Insert character s2[j-1] into s1 at position i
            ops.append(('ins', i, s2[j - 1]))
            j -= 1

    # ops are now recorded in the order "from the end of the string to the beginning".
    # This is convenient! Because if we apply the changes to the list of characters
    # from the end to the beginning, the indices of the start of the string won't "shift".

    path_strings = []
    # Add the original s1 as a starting point
    current_chars = list(s1)
    path_strings.append("".join(current_chars))

    for op in ops:
        type_ = op[0]

        if type_ == 'sub':
            idx, char = op[1], op[2]
            current_chars[idx] = char
        elif type_ == 'del':
            idx = op[1]
            del current_chars[idx]
        elif type_ == 'ins':
            idx, char = op[1], op[2]
            current_chars.insert(idx, char)

        # After each atomic operation, save the new version of the string
        path_strings.append("".join(current_chars))

    return path_strings


def levenshtein_distance_fast(s1, s2):
    return Levenshtein.distance(s1, s2)


# Main logic: Greedy merging with the best path search
def solve_greedy_path_search(strings):
    pool = strings[:]  # Copy

    # print(f"Start: {pool}")
    min_score = 1000000000
    best_candidate = None
    for i in range(len(strings)):
        score = sum(levenshtein_distance_fast(pool[i], s) for s in strings)
        if score < min_score:
            best_candidate = i
            min_score = score
    # print('Best candidate: {} Score: {}'.format(best_candidate, min_score))

    best_score = 1000000000
    while len(pool) > 1:
        # A. Find the closest pair (to merge similar things first)
        best_pair_idx = (-1, -1)
        min_pair_dist = float('inf')

        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                d = levenshtein_distance_fast(pool[i], pool[j])
                if d < min_pair_dist:
                    min_pair_dist = d
                    best_pair_idx = (i, j)

        idx1, idx2 = best_pair_idx
        s1, s2 = pool[idx1], pool[idx2]

        # B. Generate all variants of transforming s1 -> s2
        candidates = get_path_strings(s1, s2)

        # C. Find the candidate that best fits the ENTIRE current pool
        best_candidate = None
        min_total_score = float('inf')

        for cand in candidates:
            # Calculate the sum of distances to all strings in the pool
            # (including the ones we are merging now — this is important for the weight)
            score = sum(levenshtein_distance_fast(cand, s) for s in strings)

            if score < min_total_score:
                min_total_score = score
                best_candidate = cand

        # print(f"Merging '{s1}' and '{s2}'. Best on path: '{best_candidate}' (Score: {min_total_score})")

        # D. Replace the pair with the found candidate
        # Remove indices (larger first to avoid shifting the smaller one)
        pool.pop(max(idx1, idx2))
        pool.pop(min(idx1, idx2))
        pool.append(best_candidate)

    score = sum(levenshtein_distance_fast(pool[0], s) for s in strings)
    return pool[0], score


def ultra_fast_word_levenshtein(text1: str, text2: str) -> int:
    return editdistance.eval(text1.split(), text2.split())


def get_path_words_strings_ultra(text1, text2, tokens=False) -> list[str]:
    if tokens:
        words1 = list(text1)
        words2 = list(text2)
    else:
        words1 = text1.split()
        words2 = text2.split()

    # 1. Mapping words to unique Unicode characters
    vocab = {}
    s1_chars, s2_chars = [], []
    # Start with a safe non-printable Unicode block
    char_idx = 57344

    for w in words1:
        if w not in vocab:
            vocab[w] = chr(char_idx)
            char_idx += 1
        s1_chars.append(vocab[w])

    for w in words2:
        if w not in vocab:
            vocab[w] = chr(char_idx)
            char_idx += 1
        s2_chars.append(vocab[w])

    # 2. C function call (the matrix is calculated in fractions of a millisecond)
    s1_str = "".join(s1_chars)
    s2_str = "".join(s2_chars)

    # editops returns a list of the form [('replace', 0, 0), ('insert', 1, 1)]
    editops = Levenshtein.editops(s1_str, s2_str)

    # 3. Restoration (apply operations from the end to avoid shifting indices)
    current_words = words1[:]
    if tokens:
        path_strings = [tuple(current_words)]
    else:
        path_strings = [" ".join(current_words)]

    for op, i, j in reversed(editops):
        if op == 'replace':
            current_words[i] = words2[j]
        elif op == 'delete':
            del current_words[i]
        elif op == 'insert':
            current_words.insert(i, words2[j])
        if tokens:
            path_strings.append(tuple(current_words))
        else:
            path_strings.append(" ".join(current_words))

    return path_strings


def solve_greedy_path_search_word_level(strings):
    pool = strings[:]  # Copy

    min_score = 1000000000
    best_candidate = None
    for i in range(len(strings)):
        score = sum(ultra_fast_word_levenshtein(pool[i], s) for s in strings)
        if score < min_score:
            best_candidate = i
            min_score = score
    # print('Best candidate: {} Score: {}'.format(best_candidate, min_score))

    best_score = 1000000000
    while len(pool) > 1:
        # A. Find the closest pair (to merge similar things first)
        best_pair_idx = (-1, -1)
        min_pair_dist = float('inf')

        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                d = ultra_fast_word_levenshtein(pool[i], pool[j])
                if d < min_pair_dist:
                    min_pair_dist = d
                    best_pair_idx = (i, j)

        idx1, idx2 = best_pair_idx
        s1, s2 = pool[idx1], pool[idx2]

        # B. Generate all variants of transforming s1 -> s2
        candidates = get_path_words_strings_ultra(s1, s2)

        # C. Find the candidate that best fits the ENTIRE current pool
        best_candidate = None
        min_total_score = float('inf')

        for cand in candidates:
            # Calculate the sum of distances to all strings in the pool
            # (including the ones we are merging now — this is important for the weight)
            score = sum(ultra_fast_word_levenshtein(cand, s) for s in strings)

            if score < min_total_score:
                min_total_score = score
                best_candidate = cand

        # print(f"Merging '{s1}' and '{s2}'. Best on path: '{best_candidate}' (Score: {min_total_score})")

        # D. Replace the pair with the found candidate
        # Remove indices (larger first to avoid shifting the smaller one)
        pool.pop(max(idx1, idx2))
        pool.pop(min(idx1, idx2))
        pool.append(best_candidate)

    score = sum(ultra_fast_word_levenshtein(pool[0], s) for s in strings)
    return pool[0], score


def find_set_median(strings):
    """
    Finds the Set Median (the string from the array with the minimum sum of distances to the others).
    """
    if not strings:
        return None, 0

    n = len(strings)
    # Create an N x N matrix filled with zeros
    distances = [[0] * n for _ in range(n)]

    # Fill the distance matrix (using symmetry)
    for i in range(n):
        for j in range(i + 1, n):
            dist = levenshtein_distance_fast(strings[i], strings[j])
            distances[i][j] = dist
            distances[j][i] = dist  # d(A, B) == d(B, A)

    # Find the string with the minimum sum
    min_sum = float('inf')
    median_string = None

    for i in range(n):
        current_sum = sum(distances[i])
        if current_sum < min_sum:
            min_sum = current_sum
            median_string = strings[i]

    return median_string, min_sum


def find_weighted_set_median(strings, weights):
    """
    Finds the weighted Set Median.
    """
    if not strings or not weights or len(strings) != len(weights):
        raise ValueError("The lists of strings and weights must be non-empty and of the same length")

    n = len(strings)
    # Create an N x N matrix
    distances = [[0] * n for _ in range(n)]

    # Step 1: Fill the distance matrix (using symmetry for speed)
    for i in range(n):
        for j in range(i + 1, n):
            dist = levenshtein_distance_fast(strings[i], strings[j])
            distances[i][j] = dist
            distances[j][i] = dist

    # Step 2: Find the string with the minimum weighted sum
    min_weighted_sum = float('inf')
    median_string = None

    for i in range(n):
        # Calculate the sum of distances from the i-th string to all j-th strings, multiplying by the weight of the j-th
        current_weighted_sum = sum(distances[i][j] * weights[j] for j in range(n))

        if current_weighted_sum < min_weighted_sum:
            min_weighted_sum = current_weighted_sum
            median_string = strings[i]

    return median_string, min_weighted_sum


def find_weighted_set_median_extended(strings, weights):
    """
    Finds the weighted Set Median.
    """
    if not strings or not weights or len(strings) != len(weights):
        raise ValueError("The lists of strings and weights must be non-empty and of the same length")

    n = len(strings)
    all_candidates = []
    seen = set()

    def _add(c):
        if c not in seen:
            seen.add(c)
            all_candidates.append(c)

    # Create extended set of strings with all intermediate strings
    for i in range(n):
        _add(strings[i])
        for j in range(i + 1, n):
            candidates = get_path_strings(strings[i], strings[j])
            for c in candidates:
                _add(c)
    all_candidates = list(all_candidates)
    a = len(all_candidates)

    # Create an A x N matrix
    distances = np.zeros((a, n), dtype=np.int32)

    # Step 1: Fill the distance matrix
    for i in range(a):
        for j in range(n):
            dist = levenshtein_distance_fast(all_candidates[i], strings[j])
            distances[i, j] = dist

    # Step 2: Find the string with the minimum weighted sum
    min_weighted_sum = float('inf')
    median_string = None

    for i in range(a):
        # Calculate the sum of distances from the i-th string to all j-th strings, multiplying by the weight of the j-th
        current_weighted_sum = sum(distances[i, j] * weights[j] for j in range(n))

        if current_weighted_sum < min_weighted_sum:
            min_weighted_sum = current_weighted_sum
            median_string = all_candidates[i]

    return median_string, min_weighted_sum


def find_weighted_word_set_median(texts, weights):
    """
    Finds the weighted Set Median at the word level.
    """
    if not texts or not weights or len(texts) != len(weights):
        raise ValueError("The lists of texts and weights must be non-empty and of the same length")

    n = len(texts)

    # OPTIMIZATION: Pre-split all strings into words once.
    # This saves us from calling .split() inside nested loops.
    tokenized_texts = [text.split() for text in texts]

    # Create an N x N matrix
    distances = [[0] * n for _ in range(n)]

    # Step 1: Fill the distance matrix (using symmetry)
    for i in range(n):
        for j in range(i + 1, n):
            # Compare lists of words, not strings
            dist = editdistance.eval(tokenized_texts[i], tokenized_texts[j])
            distances[i][j] = dist
            distances[j][i] = dist

    # Step 2: Find the text with the minimum weighted sum
    min_weighted_sum = float('inf')
    median_text = None

    for i in range(n):
        # Calculate the sum of distances from the i-th text to all j-th texts, multiplying by the weight of the j-th
        current_weighted_sum = sum(distances[i][j] * weights[j] for j in range(n))

        if current_weighted_sum < min_weighted_sum:
            min_weighted_sum = current_weighted_sum
            # Return the original string, not the split list
            median_text = texts[i]

    return median_text, min_weighted_sum


def find_weighted_word_set_median_extended(texts, weights):
    """
    Finds the weighted Set Median at the word level.
    """
    if len(texts) != len(weights):
        raise ValueError("The lists of texts and weights must be non-empty and of the same length")

    n = len(texts)
    tokenized_texts = [tuple(text.split()) for text in texts]

    all_candidates = set([])
    # Create extended set of strings with all intermediate strings
    for i in range(n):
        all_candidates |= set([tokenized_texts[i]])
        for j in range(i + 1, n):
            candidates = get_path_words_strings_ultra(tokenized_texts[i], tokenized_texts[j], tokens=True)
            for c in candidates:
                all_candidates |= set([c])
    all_candidates = list(all_candidates)
    a = len(all_candidates)

    # Create an A x N matrix
    distances = np.zeros((a, n), dtype=np.int32)

    # Step 1: Fill the distance matrix
    for i in range(a):
        for j in range(n):
            dist = editdistance.eval(all_candidates[i], tokenized_texts[j])
            distances[i, j] = dist

    # Step 2: Find the string with the minimum weighted sum
    min_weighted_sum = float('inf')
    median_string = None

    for i in range(a):
        # Calculate the sum of distances from the i-th string to all j-th strings, multiplying by the weight of the j-th
        current_weighted_sum = sum(distances[i, j] * weights[j] for j in range(n))

        if current_weighted_sum < min_weighted_sum:
            min_weighted_sum = current_weighted_sum
            median_string = " ".join(all_candidates[i])

    return median_string, min_weighted_sum


def load_jsonl_to_dict(file_paths):
    data = {}
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                uid = item['audio']
                text = item['text']
                if uid not in data:
                    data[uid] = []
                data[uid].append(text)
    return data


def _ensemble_single_text_worker(
        index,
        data,
        weights,
        ensemble_type,
        normalize,
        normalizer,
        char_level
):
    """
    Helper function to process a single set of texts.
    Returns the index (to preserve order during multiprocessing) and the final text.
    """
    data_for_processing = []
    for d in data:
        if normalize:
            d1 = normalizer(d)
        else:
            d1 = d
        data_for_processing.append(d1)

    if char_level:
        # Char level
        if ensemble_type == 'greedy':
            final_text, _ = solve_greedy_path_search(data_for_processing)
        elif ensemble_type == 'median':
            final_text, _ = find_set_median(data_for_processing)
        elif ensemble_type == 'median_extended':
            final_text, _ = find_weighted_set_median_extended(data_for_processing, weights)
    else:
        # Word level
        if ensemble_type == 'greedy':
            final_text, _ = solve_greedy_path_search_word_level(data_for_processing)
        elif ensemble_type == 'median':
            final_text, _ = find_weighted_word_set_median(data_for_processing, weights)
        elif ensemble_type == 'median_extended':
            final_text, _ = find_weighted_word_set_median_extended(data_for_processing, weights)

    return index, final_text


def ensemble_text_lists(
        texts_list,
        normalize=True,
        char_level=True,
        weights=None,
        ensemble_type='median_extended',
        language=None,
        max_workers=8,
        verbose=True,
):
    """
    Ensembles multiple text hypotheses for a batch of samples into a single best text per sample.

    This function takes a 2D list where each inner list contains hypotheses generated by
    different models (or decoding strategies) for a single example. It computes the best
    consensus string using Levenshtein distance-based algorithms (either at the character
    or word level) and returns a 1D list of ensembled texts maintaining the original order.

    Args:
        texts_list (list of list of str): A 2D list of shape (N, M), where N is the number
            of samples and M is the number of hypotheses per sample.
            Example: [["hypo 1", "hypo 2"], ["hypo 1 for sample 2", "hypo 2 for sample 2"]].
        normalize (bool, optional): If True, applies text normalization (e.g., lowercasing,
            punctuation stripping, abbreviation expansion) to the input strings before
            ensembling. Defaults to True.
        char_level (bool, optional): If True, computes the edit distance and paths at the
            character level. If False, operates at the word level (splitting by spaces).
            Defaults to True.
        weights (list of float or int, optional): A list of length M containing weights
            for each hypothesis source. Higher weights give more importance to the
            corresponding model. If None, all hypotheses are weighted equally.
            Defaults to None.
        ensemble_type (str, optional): The strategy used to compute the consensus string.
            Supported values:
            - 'greedy': Iteratively merges pairs of strings with the shortest Levenshtein path.
            - 'median': Selects the string from the given pool that minimizes the total
              (weighted) distance to all other strings.
            - 'median_extended': Evaluates the original strings plus all intermediate
              strings on the Levenshtein paths between them, returning the optimal
              weighted set median. Defaults to 'median_extended'.
        language (str, optional): The language code to determine the normalization rules.
            If 'en', `EnglishTextNormalizer` is used. Otherwise, it defaults to a
            `BasicMultilingualTextNormalizer`. Defaults to None.
        max_workers (int, optional): The maximum number of worker processes to use
            for parallel execution. If set to 1 or lower, the processing runs sequentially
            on the main thread. Defaults to 8.
        verbose (bool, optional): If True, prints configuration details, status messages,
            and displays a progress bar via tqdm. Defaults to True.

    Raises:
        ValueError: If the `weights` list is provided but its length does not exactly
            match the number of hypotheses per sample (M).

    Returns:
        list of str: A list of length N containing the final ensembled texts. Returns
        an empty list if `texts_list` is empty.
    """
    if not texts_list:
        return []

    n_samples = len(texts_list)
    n_models = len(texts_list[0])

    if weights is None:
        weights = [1] * n_models

    if len(weights) != n_models:
        raise ValueError(
            f"Length of weights ({len(weights)}) must match the number of hypotheses per item ({n_models})")

    if verbose:
        print("Normalize: {}, Char level: {}".format(normalize, char_level))
        print("Weights: {}".format(weights))
        print("Ensemble type: {}".format(ensemble_type))

    if language == 'en':
        if verbose:
            print("Use English language text normalizer")
        normalizer = EnglishTextNormalizer(english_spelling_normalizer)
    else:
        if verbose:
            print("Use Multi-lingual text normalizer")
        normalizer = BasicMultilingualTextNormalizer()

    if verbose:
        print('Ensembling {} items with {} workers...'.format(n_samples, max_workers))

    start_time = time.time()

    # Pre-allocate an array of the required size to preserve order
    results = [None] * n_samples

    if max_workers <= 1:
        iterator = range(n_samples)
        if verbose:
            iterator = tqdm(iterator, total=n_samples)

        for i in iterator:
            _, final_text = _ensemble_single_text_worker(
                index=i,
                data=texts_list[i],
                weights=weights,
                ensemble_type=ensemble_type,
                normalize=normalize,
                normalizer=normalizer,
                char_level=char_level
            )
            results[i] = final_text
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(n_samples):
                futures.append(
                    executor.submit(
                        _ensemble_single_text_worker,
                        i,
                        texts_list[i],
                        weights,
                        ensemble_type,
                        normalize,
                        normalizer,
                        char_level
                    )
                )

            future_iterator = concurrent.futures.as_completed(futures)
            if verbose:
                future_iterator = tqdm(future_iterator, total=n_samples)

            for future in future_iterator:
                idx, final_text = future.result()
                # Restore the result at the original index
                results[idx] = final_text

    if verbose:
        print("Ensembling complete in {:.2f} seconds".format(time.time() - start_time))

    return results


def ensemble_jsonl_files(
        input_files,
        output_file,
        normalize=True,
        char_level=True,
        weights=None,
        ensemble_type='median_extended',
        language=None,
        max_workers=8,
        verbose=True,
):
    """
    Reads multiple JSONL files containing text hypotheses, ensembles them using the
    specified algorithm, and writes the resulting texts to an output JSONL file.

    This function aggregates predictions from multiple models (or multiple decoding
    strategies) for the same audio samples. It groups the texts by their 'audio' UID
    and delegates the core multiprocessing and ensembling logic to `ensemble_texts`.

    Args:
        input_files (list of str): A list of file paths to the input JSONL files.
            Each file should contain JSON objects with 'audio' (unique ID) and
            'text' (hypothesis) keys.
        output_file (str): The file path where the ensembled results will be saved.
            The output will be in JSONL format containing 'audio' and 'text' keys.
        normalize (bool, optional): If True, applies text normalization to the input
            strings before ensembling. Defaults to True.
        char_level (bool, optional): If True, computes Levenshtein distances and
            paths at the character level. If False, operates at the word level.
            Defaults to True.
        weights (list of float or int, optional): A list of weights corresponding to
            each input file. Higher weights give more importance to the respective
            model. If None, all files are weighted equally. Defaults to None.
        ensemble_type (str, optional): The ensembling strategy to use. Supported values:
            - 'greedy': Greedily merges strings based on shortest Levenshtein paths.
            - 'median': Finds the set median (the string with the minimum sum of
              distances to all other strings in the pool).
            - 'median_extended': Finds the weighted set median, also considering all
              intermediate strings on the Levenshtein paths. Defaults to 'median_extended'.
        language (str, optional): The language code used to select the text normalizer.
            If 'en', it uses `EnglishTextNormalizer`. For any other value or None,
            it defaults to `BasicMultilingualTextNormalizer`. Defaults to None.
        max_workers (int, optional): The maximum number of worker processes to use
            for parallel execution. If set to 1 or lower, execution will be sequential.
            Defaults to 8.
        verbose (bool, optional): If True, prints status messages and displays a
            progress bar (via tqdm). Defaults to True.

    Raises:
        ValueError: If a `weights` list is explicitly provided but its length does
            not exactly match the number of provided `input_files`.

    Returns:
        None: The function writes the output directly to the specified `output_file`
        and does not return any value.
    """
    if weights is None:
        weights = [1] * len(input_files)

    if verbose:
        print(f"Loading data from {len(input_files)} files...")

    # Load data from all files into a dictionary grouped by audio UID
    merged_data = load_jsonl_to_dict(input_files)
    uids = list(merged_data.keys())

    # Prepare the list of lists of strings required by ensemble_texts
    texts_list = [merged_data[uid] for uid in uids]

    # Delegate the actual ensembling logic and parallel processing to ensemble_texts
    ensembled_results = ensemble_text_lists(
        texts_list=texts_list,
        normalize=normalize,
        char_level=char_level,
        weights=weights,
        ensemble_type=ensemble_type,
        language=language,
        max_workers=max_workers,
        verbose=verbose
    )

    if verbose:
        print(f"Writing results to {output_file}...")

    # Write the results matching each UID to its corresponding ensembled text
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for uid, final_text in zip(uids, ensembled_results):
            result = {
                "audio": uid,
                "text": final_text,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')

    if verbose:
        print("File writing complete.")