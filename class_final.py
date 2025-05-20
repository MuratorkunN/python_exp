class Orkun():
    def __init__(self, field_of_study: str, hours_studied: float, is_handsome: bool):
        self.field, self.hours = field_of_study, hours_studied, 
        self.__handsome = is_handsome #private :O
        
    def __le__(self, orkun2):
        if (self.__handsome and not orkun2.__handsome) or (self.__handsome <= orkun2.__handsome):
            return self.__handsome <= orkun2.__handsome
        else:
            return self.hours_studied <= orkun2.hours_studied
        
    def __gt__(self, orkun2):
        return not self.__le__(orkun2)
        
        
        

murat = Orkun("data", 0.1, 1)
orkun = Orkun("ai", 95, 0)

print(murat <= orkun)
print(murat > orkun)

#print(murat.__handsome)
print(murat._Orkun__handsome)
print(dir(orkun))