#coding=UTF_8
import pygame,sys,time
#from pygame.locals import *  设置窗口全屏，固定值等
from random import randint
class TankMain():
    width=600
    height=500
    mytank=None
    wall=None
    mytank_missile_list = []
   # enemy_list = []
    enemy_list = pygame.sprite.Group() #敌方坦克的组群
    explode_list=[]
    enemy_missile_list=pygame.sprite.Group()
    def startgame(self):
        pygame.init()#pygame的初始化，加载系统是资源
        #设置一个窗口（600,500）代表窗口的长和宽，0是默认值，固定值（0，RESIZEBLE,FULLSCREEM）,32是颜色值，创建窗口
        screem=pygame.display.set_mode((TankMain.width,TankMain.height),0,32)
        #给窗口设置标题
        pygame.display.set_caption("坦克大战")
        TankMain.mytank=Mytank(screem)
        TankMain.wall=Wall(screem,65,200,30,120)#30是宽，120是长度
        if len(TankMain.enemy_list)==0:

            for i in range(1,6):
                TankMain.enemy_list.add(Enemytank(screem))#把敌方坦克放到一个组里



        while True:
            if len(TankMain.enemy_list) < 5:
                TankMain.enemy_list.add(Enemytank(screem))
            #设置屏幕颜色，RGB(0,0，0)是黑色，（255，255,255）是白色
            screem.fill((0,0,0))
            for i,text in enumerate(self.write_text(),0):#枚举，第一个参数是一个数组，第二个参数是从第一个开始

            #在某一个图像上画一个新的图
                screem.blit(text,(0,5+(15*i)))
            #显示游戏中的墙,并且对墙和其他对象进行碰撞检测
            TankMain.wall.display()
            TankMain.wall.hit_others()
            self.get_event(TankMain.mytank,screem)  #获取事件，根据不同的选择来进行相应的操作
            if TankMain.mytank:
                TankMain.mytank.hit_enemy_missile()#我方的坦克与敌方的炮弹进行碰撞检测
            if TankMain.mytank and  TankMain.mytank.live:

                TankMain.mytank.display()
                TankMain.mytank.hit_enemy_tank()
            #显示重置
                TankMain.mytank.move()
            else:
                TankMain.mytank=None
            for enemy in TankMain.enemy_list:
                enemy.display()
                enemy.random_move()
                enemy.random_fire()

            for m in TankMain.mytank_missile_list:
                if m.live:
                    m.display()
                    m.hit_tank()#炮弹打中敌方坦克
                    m.move()
                else:
                    TankMain.mytank_missile_list.remove(m)
            for m in TankMain.enemy_missile_list:
                if m.live:
                    m.display()
                    m.move()
                else:
                    TankMain.enemy_missile_list.remove(m)
            for explode in TankMain.explode_list:
                explode.display()

            time.sleep(0.05)#每次休眠0.05秒跳到下一帧
            pygame.display.update()

    #获取所有的事件（键盘敲击，鼠标点击）
    def get_event(self,mytank,screem):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stopgame()
            if event.type==pygame.KEYDOWN and  (not mytank) and event.key==pygame.K_n:
                TankMain.mytank=Mytank(screem)
            if event.type==pygame.KEYDOWN and mytank:
                if event.key==pygame.K_LEFT or event.key==pygame.K_a:
                    mytank.direction="L"
                    mytank.stop=False
                    # mytank.move()
                if  event.key==pygame.K_RIGHT or event.key==pygame.K_d:
                    mytank.direction = "R"
                    mytank.stop = False
                    # mytank.move()
                if event.key==pygame.K_UP or event.key==pygame.K_w:
                    mytank.direction = "U"
                    mytank.stop = False
                    # mytank.move()
                if event.key==pygame.K_DOWN or event.key==pygame.K_s:
                    mytank.direction = "D"
                    mytank.stop = False
                    # mytank.move()
                if event.key==pygame.K_ESCAPE:#敲击esc键也是退出
                    self.stopgame()
                if event.key==pygame.K_SPACE:
                    m=mytank.fire()
                    m.good=True #我方坦克发射的炮道是好炮弹
                    TankMain.mytank_missile_list.append(m)
            if event.type==pygame.KEYUP and mytank:
                if event.key==pygame.K_LEFT or event.key==pygame.K_RIGHT or event.key==pygame.K_UP or event.key==pygame.K_DOWN:
                    mytank.stop=True


    def stopgame(self):
        sys.exit()#系统推出
    def write_text(self):
        #屏幕中内容的设置，颜色字体等设置,定义一个字体
        font=pygame.font.SysFont("simsunnsimsun",12 )
        #根据字体来设置内容，创建文字的图像，屏幕中所有都是sf
        test_sf1=font.render("敌方坦克数量：%d"%len(TankMain.enemy_list),True,(255,0,0))
        test_sf2=font.render("我方坦克炮弹数量：%d"%len(TankMain.mytank_missile_list),True,(255,0,0))
        return test_sf1,test_sf2
#坦克大战游戏中所有对象的父类，父类是pygame.sprite.Sprite
class BaseItem(pygame.sprite.Sprite):
    def __init__(self,screem):
        pygame.sprite.Sprite.__init__(self)  #自定义的子类中必须要继承的父类的init方法
        self.screem=screem
    #炮弹显示
    def display(self):
        if self.live:
            self.image=self.images[self.direction]
            self.screem.blit(self.image,self.rect)
class Tank(BaseItem):
    #所有的坦克对象高度和宽度都是一样的，永远不变
    width=50
    height=50
    def __init__(self,screem,left,top):
        super().__init__(screem)#继承父类的init方法
        self.direction="L"#默认值
        self.speed=8#坦克移动的速度
        self.stop=False
        self.images={}#坦克的所有图片，key：方向，value：图片（surface）
        self.images["L"]=pygame.image.load("images2/tankL.jpg")
        self.images["R"]=pygame.image.load("images2/tankR.jpg")
        self.images["U"]=pygame.image.load("images2/tankU.jpg")
        self.images["D"]=pygame.image.load("images2/tankD.jpg")
        self.image=self.images[self.direction]#坦克的图片由方向决定
        self.rect=self.image.get_rect()#获取图片的边界
        self.rect.left=left
        self.rect.top=top
        self.good=False

        self.live=True#决定坦克是否消灭
        self.oldleft = left
        self.oldtop = top

     #把坦克图片显示在窗口上
    def stay(self):
        self.rect.top=self.oldtop
        self.rect.left=self.oldleft

    def move(self):
        if not self.stop:#如果坦克不是停止状态
            self.oldleft=self.rect.left
            self.oldtop=self.rect.top
            if self.direction=="L":
                if self.rect.left>0:#判断坦克是否在边界的左边
                    self.rect.left-=self.speed
                else:
                    self.rect.left=0
            elif  self.direction=="R":
                if self.rect.right<TankMain.width:#判断坦克是否在边界的左边
                    self.rect.right+=self.speed
                else:
                    self.rect.right=TankMain.width
            elif  self.direction=="U":
                if self.rect.top>0:#判断坦克是否在边界的左边
                    self.rect.top-=self.speed
                else:
                    self.rect.top=0
            elif  self.direction=="D":
                if self.rect.bottom<TankMain.height:#判断坦克是否在边界的左边
                    self.rect.bottom+=self.speed
                else:
                    self.rect.bottom=TankMain.height
    def fire(self):
        m=Missile(self.screem,self)
        return m

class Mytank(Tank):
    def __init__(self,screem):
        super().__init__(screem,275,400) #创建我方坦克，坦克显示在屏幕的中下 的位置
        self.stop=True
        self.live=True
    def hit_enemy_missile(self):
        hit_list=pygame.sprite.spritecollide(self,TankMain.enemy_missile_list,False)
        for m in hit_list:#我方坦克中弹了
            m.live=False
            TankMain.enemy_missile_list.remove(m)
            self.live=False
            explode=Explode(self.screem,self.rect)
            TankMain.explode_list.append(explode)
    #我方坦克与敌方坦克进行碰撞时，我方坦克发生爆炸并消失
    def hit_enemy_tank(self):
        mytank_enemytank_list=pygame.sprite.spritecollide(self,TankMain.enemy_list,False)#将我方坦克与敌方坦克进行一一比对碰撞，返回的是碰撞到的大方坦克，放在一个列表中
        for m in mytank_enemytank_list:
            self.live=False  #我方坦克消失
            explode = Explode(self.screem, self.rect)  #伴随爆炸的效果
            explode.display()
class Enemytank(Tank):


    def __init__(self,screem):
        super().__init__(screem,randint(1,5)*100,200)
        self.step=8#坦克按照方向移动的连续步数
        self.get_random_direction()
        self.speed=4

    def  get_random_direction(self):
        r = randint(0, 4)  # 得到一个坦克移动方向和停止的随机数
        if r == 4:
            self.stop = True
        elif r == 1:
            self.direction = "L"
            self.stop=False
        elif r == 2:
            self.direction = "R"
            self.stop = False
        elif r == 3:
            self.direction = "U"
            self.stop = False
        elif r == 0:
            self.direction = "D"
            self.stop = False

    #敌方坦克，按照一个确定随机方向，连续移动6步，才能换方向移动
    def random_move(self):
        if self.live:
            # tank_hit_list=pygame.sprite.spritecollide(self,TankMain.enemy_list,False)
            # for k in tank_hit_list:
            #     k.get_random_direction()
            #     k.step=6
            if self.step==0:
                self.get_random_direction()
                self.step=6
            else:
                self.move()
                self.step-=1



    def random_fire(self):
        r=randint(1,50)
        if r==10:
            m=self.fire()
            TankMain.enemy_missile_list.add(m)
        else:
            return

class Missile(BaseItem):
    width=12
    height=12

    def __init__(self,screem,tank):
        super().__init__(screem)
        self.direction = tank.direction  # 炮弹的方向是由所发射的坦克方向所决定
        self.speed = 12 # 炮弹移动的速度
        self.stop = False
        self.images = {}  # 炮弹的所有图片，key：方向，value：图片（surface）
        self.images["L"] = pygame.image.load("images2/missileL.png")
        self.images["R"] = pygame.image.load("images2/missileR.png")
        self.images["U"] = pygame.image.load("images2/missileU.png")
        self.images["D"] = pygame.image.load("images2/missileD.png ")
        self.image = self.images[self.direction]  # 坦克的图片由方向决定
        self.rect = self.image.get_rect()
        self.rect.left = tank.rect.left+(tank.width-self.width)/2
        self.rect.top =tank.rect.top+(tank.height-self.height)/2
        self.live = True  # 决定炮弹是否消灭

    def move(self):
        if self.live:#如果炮弹是活的，存在的
            if self.direction=="L":
                if self.rect.left>0:#判断坦克是否在边界的左边
                    self.rect.left-=self.speed
                else:
                    self.live=False
            elif  self.direction=="R":
                if self.rect.right<TankMain.width:#判断坦克是否在边界的左边
                    self.rect.right+=self.speed
                else:
                    self.live = False
            elif  self.direction=="U":
                if self.rect.top>0:#判断坦克是否在边界的左边
                    self.rect.top-=self.speed
                else:
                    self.live = False
            elif  self.direction=="D":
                if self.rect.bottom<TankMain.height:#判断坦克是否在边界的左边
                    self.rect.bottom+=self.speed
                else:
                    self.live = False
        #炮弹击中坦克，第一种是我方炮弹击中敌方坦克，一种是敌方炮弹击中我方坦克
    def hit_tank(self):
        if self.good:#如果炮弹是我方炮弹
            hit_list=pygame.sprite.spritecollide(self,TankMain.enemy_list,False)
            for e in  hit_list:
                e.live=False
                TankMain.enemy_list.remove(e) #如果敌方坦克被击中，从类表中删除敌方坦克
                self.live=False
                explode=Explode(self.screem,e.rect)
                TankMain.explode_list.append(explode)




class Explode(BaseItem):
    def __init__(self,screem,rect):
        super().__init__(screem)
        self.live=True
        self.images=[pygame.image.load("images2/0.gif"),\
                     pygame.image.load("images2/1.gif"),\
                     pygame.image.load("images2/2.gif"),\
                     pygame.image.load("images2/3.gif"),\
                     pygame.image.load("images2/4.gif"),\
                     pygame.image.load("images2/5.gif")]  #图片的显示，是由几个图片依次显示，所以只需要列表即可，可以用\来转义
        self.step=0
        self.rect=rect  #爆炸的位置和发生爆炸的坦克的位置一样，在构建爆炸的时候吧坦克的rect传递进来
    #display方法在整个游戏中循环调用，每隔0.05秒调用一次
    def display(self):
        if self.live:
            if self.step==len(self.images):#意味着最后一张图片已经显示
                self.live=False
            else:
                self.image=self.images[self.step]
                self.screem.blit(self.image,self.rect)
                self.step+=1
        else:
            pass#删除该对象


#游戏中的障碍物
class Wall(BaseItem):
    def __init__(self,screem ,left,top,width,height):
        super().__init__(screem)
        self.rect=pygame.Rect(left,top,width,height)  #构建出一个矩形，有长宽高
        self.color=(255,0,0)
    def display(self):

        self.screem.fill(self.color,self.rect)#这里不能用blit方法，因为他是一个将图像画在桌面上
    def hit_others(self):  #针对墙和其他坦克或者炮弹的碰撞检测
        if TankMain.mytank:
            is_hit=pygame.sprite.collide_rect(self,TankMain.mytank)
            if is_hit:
                TankMain.mytank.stop=True
                TankMain.mytank.stay()
        if TankMain.enemy_list:
            hit_list=pygame.sprite.spritecollide(self,TankMain.enemy_list,False)
            for e in hit_list:
                e.stop=True
                e.stay()
        missile_list=list(TankMain.enemy_missile_list)+TankMain.mytank_missile_list
        if missile_list:
            hit_missile_list=pygame.sprite.spritecollide(self,missile_list,False)
            for h in hit_missile_list:
                h.live=False
game=TankMain()
game.startgame()
