class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        nums_dict ={}
        for i, num in enumerate(nums):
            if num in nums_dict:
                return [nums_dict[num],i]
            else:
                nums_dict[target-num] = i