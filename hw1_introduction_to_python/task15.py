from typing import List


def hello(x = ''):
    if not x:
        return 'Hello!'
    else:
        return f'Hello, {x}!'

def int_to_roman(x):
    roman_num = {1000: 'M', 900: 'CM', 500: 'D', 400: 'CD', 
                 100: 'C', 90: 'XC', 50: 'L', 40: 'XL', 
                 10: 'X', 9: 'IX', 5: 'V', 4: 'IV', 1: 'I'}
    res = ''
    for key in roman_num:
        while x >= key:
            x -= key
            res += roman_num[key]
    return res

def longest_common_prefix(x):
    res_str = ''
    if not x:
        return res_str
    min_len = len(x[0])
    x_new = []
    for i in range(len(x)):
        x_new.append(x[i].lstrip(' \n\t'))
        min_len = min(min_len, len(x_new[i]))
    for i in range(min_len):
        for s in x_new:
            if i == len(s) or s[i] != x_new[0][i]:
                return res_str
        res_str += x_new[0][i]
    return res_str

def primes():
    num = 2
    while True:
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                break
        else:
            yield num
        num += 1

class BankCard:
    def __init__(self, total_sum, balance_limit=None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit
    
    def __call__(self, sum_spent):
        if sum_spent > self.total_sum:
            raise ValueError('Not enough money to spend sum_spent dollars.')
        else:
            self.total_sum -= sum_spent
            return f'You spent {sum_spent} dollars.'
    
    def __str__(self):
        return 'To learn the balance call balance.'
    
    def __add__(self, other):
        if other.balance_limit is None:
            return BankCard(self.total_sum + other.total_sum, other.balance_limit)
        if self.balance_limit is None or self.balance_limit > other.balance_limit:
            return BankCard(self.total_sum + other.total_sum, self.balance_limit)
        return BankCard(self.total_sum + other.total_sum, other.balance_limit)
            
    @property
    def balance(self):
        if self.balance_limit is not None and self.balance_limit <= 0:
            raise ValueError('Balance check limits exceeded.')
        elif self.balance_limit is not None:
            self.balance_limit -= 1
        return self.total_sum
    
    def put(self, sum_put):
        self.total_sum += sum_put
        print(f'You put {sum_put} dollars.')
      