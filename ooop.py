a = [1, 2, 3]
a_list = list(a)
a_index = a[:]
a_copy = a.copy()
a.append(5)
print("Shallow copy")
print("a_list: ", a_list)
print("a_index: ", a_index)
print("a_copy: ", a_copy)
