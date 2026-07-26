# def numbers():

#     print("Start")

#     yield 1

#     print("Middle")

#     yield 2

#     print("End")

#     yield 3
# gen = numbers()
# print(next(gen))    
# print(next(gen))    

def numbers():
    print("Start")
    return 1
    print("End")

print(numbers())
print(numbers())    