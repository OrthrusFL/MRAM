import logging


def evaluate(test_result):

    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    map_sum = 0
    mrr_sum = 0

    no_truth_num = 0
    commit_ids = test_result.drop_duplicates(subset=['commit_id'], keep='first')['commit_id'].to_list()
    for commit_id in commit_ids:
        temp = test_result[test_result['commit_id'] == commit_id]
        temp_sorted = temp.sort_values('fault_prob', ascending=False)
        temp_sorted = temp_sorted.reset_index()
        truth_ranked = temp_sorted[temp_sorted['label'] == 1].index.tolist()
        # logger.info('commit id:{}, ranking result:{}'.format(commit_id, truth_ranked))

        if len(truth_ranked) == 0:
            no_truth_num += 1
            continue
        min_rank = min(truth_ranked)  # begin from zero
        # logger.info('best result:{}'.format(min_rank))
        if (min_rank + 1) <= 10:
            hit_10 += 1
        if (min_rank + 1) <= 5:
            hit_5 += 1
        if min_rank == 0:
            hit_1 += 1

        # map
        p = 0
        length = len(truth_ranked)
        for j in range(0, length):
            p = p + ((j+1) / (truth_ranked[j]+1))
        ap = p / length
        map_sum += ap

        # mrr
        best_rank = truth_ranked[0]
        reciprocal_rank = 1 / (best_rank + 1)
        mrr_sum += reciprocal_rank

    hit_1_percent = hit_1 / (len(commit_ids) - no_truth_num)
    hit_5_percent = hit_5 / (len(commit_ids) - no_truth_num)
    hit_10_percent = hit_10 / (len(commit_ids) - no_truth_num)
    map_result = map_sum / (len(commit_ids) - no_truth_num)
    mrr_result = mrr_sum / (len(commit_ids) - no_truth_num)

    return hit_1_percent, hit_5_percent, hit_10_percent, map_result, mrr_result

