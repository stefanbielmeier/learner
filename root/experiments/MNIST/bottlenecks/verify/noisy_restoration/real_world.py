
zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = get_first_fifty_images(inBinary=True)

bottleneck0s = get_bottleneck_idxs(zeros)[0]
bottleneck1s = get_bottleneck_idxs(ones)[0]
mem_ones = get_memorization_capacity(ones, recall_quality = 1.0, startAt = 38, test_idxs = bottleneck1s) 