class Enemy:
    def __init__(self, name : str, life : float):
        self.name, self.life = name, life
        
    def reduce_life(self, float):
        self.life -= float

class Weapon:
    def __init__(self, name : str, hit : float=100):
        self.name, self.hit = name, hit
        self.use_count = 0
        
    def display_hit(self):
        print(self.hit)
        
    def display_use_count(self):
        print(self.use_count)
    
        
class Sword(Weapon):
    def __init__(self, name : str, hit : float=100):
        super(name, hit)
        self.enchanted = False
        
    def enchant(self):
        if not self.enchanted:  
            self.enchanted = True
            self.hit *= 1.5
    
    def use(self, Enemy : Enemy):
        if self.enchanted:
            Enemy.reduce_life(self.hit * 1.5)
        else:
            Enemy.reduce_life(self.hit)
        self.use_count += 1
        
            
            

        

