import csv

with open('test.log', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 스킵
    times = [int(row[0]) for row in reader]
intervals = [(times[i+1] - times[i]) / 1e9 for i in range(len(times)-1)]  # 초 단위
print('샘플 수:', len(times))
print('평균 간격:', sum(intervals)/len(intervals), '초')
print('목표 간격: 0.05 초')
print('오차:', abs(sum(intervals)/len(intervals) - 0.05) / 0.05 * 100, '%')
