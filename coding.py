nums = [3, 2, 1, 0, 4]

lastPos = 0

for i in range(len(nums)):
    if i > lastPos:
        print(False)
    lastPos = max(lastPos, i+nums[i])
print(True)
