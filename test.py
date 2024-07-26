class Solution:
    def twoSum(self, nums: List[int], target: int, i = 0) -> List[int]:
        for j in range(len(nums)):
            if i!=j:
                if nums[i]+nums[j] == target:
                    return [i,j]
        i+=1
        twoSum(nums, target, i)
        
        