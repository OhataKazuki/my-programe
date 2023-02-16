#!/usr/bin/env python
# coding: utf-8

# 複数のロボットで臭気源を特定するシミュレーション
# 物質量の多い場所を特定するようにプログラムした

# ライブラリのインポート

# In[35]:


import matplotlib
matplotlib.use('nbagg')
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np
import scipy.interpolate as interp


# イテラブルとして使用できるスロットを持つオブジェクトの基本クラス。

# In[36]:


class SlottedIterable(object):

    __slots__ = ()

    def __iter__(self):
        """Iterate through slot attributes in defined order."""
        for name in self.__slots__:
            yield getattr(self, name)

    def __repr__(self):
        """String representation of object."""
        return '{cls}({attr})'.format(
            cls=self.__class__.__name__,
            attr=', '.join(['{0}={1}'.format(
                name, getattr(self, name)) for name in self.__slots__]))


# 単一の匂いパフのプロパティのコンテナ。

# In[37]:


class Puff(SlottedIterable):
    __slots__ = ('x', 'y', 'z', 'r_sq')

    def __init__(self, x, y, z, r_sq):
        assert r_sq >= 0., 'r_sq must be non-negative.'
        self.x = x
        self.y = y
        self.z = z
        self.r_sq = r_sq


# 軸に沿った長方形の領域。

# In[38]:


class Rectangle(SlottedIterable):

    __slots__ = ('x_min', 'x_max', 'y_min', 'y_max')

    def __init__(self, x_min, x_max, y_min, y_max):
        assert x_min < x_max, 'Rectangle x_min must be < x_max.'
        assert y_min < y_max, 'Rectangle y_min must be < y_max.'
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    @property
    def w(self):
        return self.x_max - self.x_min

    @property
    def h(self):
        return self.y_max - self.y_min

    def contains(self, x, y):
        return (x >= self.x_min and x <= self.x_max and
                y >= self.y_min and y <= self.y_max)


# Farrell et. al. (2002) によるパフベースの臭気プルーム分散モデル。

# In[39]:


class PlumeModel(object):
    def __init__(self, sim_region=None, source_pos=(5., 0., 0.),
                 wind_model=None, model_z_disp=True, centre_rel_diff_scale=2.,
                 puff_init_rad=0.0316, puff_spread_rate=0.001,
                 puff_release_rate=10, init_num_puffs=10, max_num_puffs=1000,
                 rng=None):
        if sim_region is None:
            sim_region = Rectangle(0, 50., -12.5, 12.5)
        if rng is None:
            rng = np.random
        self.sim_region = sim_region
        if wind_model is None:
            wind_model = WindModel()
        self.wind_model = wind_model
        self.rng = rng
        self.model_z_disp = model_z_disp
        self._vel_dim = 3 if model_z_disp else 2
        if model_z_disp and hasattr(centre_rel_diff_scale, '__len__'):
            assert len(centre_rel_diff_scale) == 2, (
                'When model_z_disp=True, centre_rel_diff_scale must be a '
                'scalar or length 1 or 3 iterable.')
        self.centre_rel_diff_scale = centre_rel_diff_scale
        assert sim_region.contains(source_pos[0], source_pos[1]), (
            'Specified source position must be within simulation region.')
        # default to zero height source when source_pos is 2D
        source_z = 0 if len(source_pos) != 3 else source_pos[2]
        self._new_puff_params = (
            source_pos[0], source_pos[1], source_z, puff_init_rad**2)
        self.puff_spread_rate = puff_spread_rate
        self.puff_release_rate = puff_release_rate
        self.max_num_puffs = max_num_puffs
        # initialise puff list with specified number of new puffs
        self.puffs = [
            Puff(*self._new_puff_params) for i in range(init_num_puffs)]

    def update(self, dt):
        # add more puffs (stochastically) if enough capacity
        if len(self.puffs) < self.max_num_puffs:
            # puff release modelled as Poisson process at fixed mean rate
            # with number to release clipped if it would otherwise exceed
            # the maximum allowed
            num_to_release = min(
                self.rng.poisson(self.puff_release_rate * dt),
                self.max_num_puffs - len(self.puffs))
            self.puffs += [
                Puff(*self._new_puff_params) for i in range(num_to_release)]
        # initialise empty list for puffs that have not left simulation area
        alive_puffs = []
        for puff in self.puffs:
            # interpolate wind velocity at Puff position from wind model grid
            # assuming zero wind speed in vertical direction if modelling
            # z direction dispersion
            wind_vel = np.zeros(self._vel_dim)
            wind_vel[:2] = self.wind_model.velocity_at_pos(puff.x, puff.y)
            # approximate centre-line relative puff transport velocity
            # component as being a (Gaussian) white noise process scaled by
            # constants
            filament_diff_vel = (self.rng.normal(size=self._vel_dim) *
                                 self.centre_rel_diff_scale)
            vel = wind_vel + filament_diff_vel
            # update puff position using Euler integration
            puff.x += vel[0] * dt
            puff.y += vel[1] * dt
            if self.model_z_disp:
                puff.z += vel[2] * dt
            # update puff size using Euler integration with second puff
            # growth model described in paper
            puff.r_sq += self.puff_spread_rate * dt
            # only keep puff alive if it is still in the simulated region
            if puff.r_sq < 0.25:
                if self.sim_region.contains(puff.x, puff.y):
                    alive_puffs.append(puff)
        # store alive puffs only
        self.puffs = alive_puffs

    @property
    def puff_array(self):
        return np.array([tuple(puff) for puff in self.puffs])
        


# 臭気の移流を計算するための風速モデル。

# In[40]:


class WindModel(object):

    def __init__(self, sim_region=None, n_x=21, n_y=21, u_av=1., v_av=0.,
                 k_x=20., k_y=20., noise_gain=2., noise_damp=0.1,
                 noise_bandwidth=0.2, use_original_noise_updates=False,
                 rng=None):
        if sim_region is None:
            sim_region = Rectangle(0, 100, -50, 50)
        if rng is None:
            rng = np.random
        self.sim_region = sim_region
        self.u_av = u_av
        self.v_av = v_av
        self.n_x = n_x
        self.n_y = n_y
        self.k_x = k_x
        self.k_y = k_y
        # set coloured noise generator for applying boundary condition
        # need to generate coloured noise samples at four corners of boundary
        # for both components of the wind velocity field so (2,8) state
        # vector (2 as state includes first derivative)
        self.noise_gen = ColouredNoiseGenerator(
            np.zeros((2, 8)), noise_damp, noise_bandwidth, noise_gain,
            use_original_noise_updates, rng)
        # compute grid node spacing
        self.dx = sim_region.w / (n_x - 1)  # x grid point spacing
        self.dy = sim_region.h / (n_y - 1)  # y grid point spacing
        # initialise wind velocity field to mean values
        # +2s are to account for boundary grid points
        self._u = np.ones((n_x + 2, n_y + 2)) * u_av
        self._v = np.ones((n_x + 2, n_y + 2)) * v_av
        # create views on to field interiors (i.e. not including boundaries)
        # for notational ease - note this does not copy any data
        self._u_int = self._u[1:-1, 1:-1]
        self._v_int = self._v[1:-1, 1:-1]
        # preassign array of corner means values
        self._corner_means = np.array([u_av, v_av]).repeat(4)
        # precompute linear ramp arrays with size of boundary edges for
        # linear interpolation of corner values
        self._ramp_x = np.linspace(0., 1., n_x + 2)
        self._ramp_y = np.linspace(0., 1., n_y + 2)
        # set up cubic spline interpolator for calculating off-grid wind
        # velocity field values
        self._x_points = np.linspace(sim_region.x_min, sim_region.x_max, n_x)
        self._y_points = np.linspace(sim_region.y_min, sim_region.y_max, n_y)
        # initialise flag to indicate velocity field interpolators not set
        self._interp_set = True

    def _set_interpolators(self):
        self._interp_u = interp.RectBivariateSpline(
            self.x_points, self.y_points, self._u_int)
        self._interp_v = interp.RectBivariateSpline(
            self.x_points, self.y_points, self._v_int)
        self._interp_set = True

    @property
    def x_points(self):
        return self._x_points

    @property
    def y_points(self):
        return self._y_points

    @property
    def velocity_field(self):
        return np.dstack((self._u_int, self._v_int))

    def velocity_at_pos(self, x, y):
        if not self._interp_set:
            self._set_interpolators()
        return np.array([float(self._interp_u(x, y)),
                         float(self._interp_v(x, y))])

    def draw(self, ax, elems):
        pass
    
    def one_step(self, dt):
        # update boundary values
        self._apply_boundary_conditions(dt)
        # approximate spatial first derivatives with centred finite difference
        # equations for both components of wind field
        du_dx, du_dy = self._centred_first_diffs(self._u)
        dv_dx, dv_dy = self._centred_first_diffs(self._v)
        # approximate spatial second derivatives with centred finite difference
        # equations for both components of wind field
        d2u_dx2, d2u_dy2 = self._centred_second_diffs(self._u)
        d2v_dx2, d2v_dy2 = self._centred_second_diffs(self._v)
        # compute approximate time derivatives across simulation region
        # interior from defining PDEs
        #     du/dt = -(u*du/dx + v*du/dy) + 0.5*k_x*d2u/dx2 + 0.5*k_y*d2u/dy2
        #     dv/dt = -(u*dv/dx + v*dv/dy) + 0.5*k_x*d2v/dx2 + 0.5*k_y*d2v/dy2
        du_dt = (-self._u_int * du_dx - self._v_int * du_dy +
                 0.5 * self.k_x * d2u_dx2 + 0.5 * self.k_y * d2u_dy2)
        dv_dt = (-self._u_int * dv_dx - self._v_int * dv_dy +
                 0.5 * self.k_x * d2v_dx2 + 0.5 * self.k_y * d2v_dy2)
        # perform update with Euler integration
        self._u_int += du_dt * dt
        self._v_int += dv_dt * dt
        # set flag to indicate interpolators no longer valid as fields updated
        self._interp_set = False

    def _apply_boundary_conditions(self, dt):
        """Applies boundary conditions to wind velocity field."""
        # update coloured noise generator
        self.noise_gen.update(dt)
        # extract four corner values for each of u and v fields as component
        # mean plus current noise generator output
        (u_tl, u_tr, u_bl, u_br, v_tl, v_tr, v_bl, v_br) = (
            self.noise_gen.output + self._corner_means)
        # linearly interpolate along edges
        self._u[:, 0] = u_tl + self._ramp_x * (u_tr - u_tl)  # u top edge
        self._u[:, -1] = u_bl + self._ramp_x * (u_br - u_bl)  # u bottom edge
        self._u[0, :] = u_tl + self._ramp_y * (u_bl - u_tl)  # u left edge
        self._u[-1, :] = u_tr + self._ramp_y * (u_br - u_tr)  # u right edge
        self._v[:, 0] = v_tl + self._ramp_x * (v_tr - v_tl)  # v top edge
        self._v[:, -1] = v_bl + self._ramp_x * (v_br - v_bl)  # v bottom edge
        self._v[0, :] = v_tl + self._ramp_y * (v_bl - v_tl)  # v left edge
        self._v[-1, :] = v_tr + self._ramp_y * (v_br - v_tr)  # v right edge

    def _centred_first_diffs(self, f):
        return ((f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * self.dx),
                (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * self.dy))

    def _centred_second_diffs(self, f):
        return (
            (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[0:-2, 1:-1]) / self.dx**2,
            (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, 0:-2]) / self.dy**2)


# 色付き (相関) ガウス ノイズ プロセスのジェネレーター。

# In[41]:


class ColouredNoiseGenerator(object):

    def __init__(self, init_state, damping=0.1, bandwidth=0.2, gain=1.,
                 use_original_updates=False,  rng=None):
        if rng is None:
            rng = np.random
        # set up state space matrices
        self.a_mtx = np.array([
            [0., 1.], [-bandwidth**2, -2. * damping * bandwidth]])
        self.b_mtx = np.array([[0.], [gain * bandwidth**2]])
        # initialise state
        self.state = init_state
        self.rng = rng
        self.use_original_updates = use_original_updates

    @property
    def output(self):
        return self.state[0, :]

    def update(self, dt):
        # get normal random input
        n = self.rng.normal(size=(1, self.state.shape[1]))
        if self.use_original_updates:
            # apply Farrell et al. (2002) update
            self.state += dt * (self.a_mtx.dot(self.state) + self.b_mtx.dot(n))
        else:
            # apply update with Euler-Maruyama integration
            self.state += (
                dt * self.a_mtx.dot(self.state) + self.b_mtx.dot(n) * dt**0.5)


# 画面に表示する世界の状態を表すクラス

# In[42]:


class World:
    def __init__(self, time_span, time_interval, area_min, area_max, debug=True):
        self.objects = []  
        self.debug = debug
        self.time_span = time_span  
        self.time_interval = time_interval
        self.area_min = area_min
        self.area_max = area_max
        self.agents = []
        self.maps = []
        
    #要素を追加する関数
    def append(self,obj):  
        self.objects.append(obj)
        #ロボットとゴールは、別に認識
        if isinstance(obj, Agent): self.agents.append(obj)
        if isinstance(obj, Map): self.maps.append(obj)
    
    #描画関数
    def draw(self):
        #描画設定
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')             
        ax.set_xlim(int(area_min),int(area_max))                  
        ax.set_ylim(int(area_min),int(area_max)) 
        ax.set_xlabel("X",fontsize=10)                 
        ax.set_ylabel("Y",fontsize=10)
        
        #描画要素のリスト
        elems = []
        
        #描画実行（デバッグ用に分岐あり）
        if self.debug:        
            for i in range(int(self.time_span/self.time_interval)): self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                     frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            #self.ani.save('odor_sensing_abc_tripl.gif', dpi=100, fps=10 ,writer='pillow')
            plt.show()
    
    #状態を更新する関数
    def one_step(self, i, elems, ax):
        
        """"
        if len(self.maps[0].odors) < 1:
            return
        """
        
        while elems: elems.pop().remove()
        #時間を進める
        time_str = "t = %.2f[s]" % (self.time_interval*i)
        elems.append(ax.text(0, 1, time_str, fontsize=15))
        #状態更新
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)
        #ゴール判定
        for a in self.agents:
            if not a.win:
                a.time = (self.time_interval*(i + 1))
        

# ロボット本体のクラス

# In[44]:


class Robot:   
    def __init__(self, pose, wind_model, color="black"):   
        self.pose = pose
        self.r = 0.1  
        self.color = color 
        self.poses = [pose]
        self.wind_model = wind_model
    
    #描画
    def draw(self, ax, elems): 
        x, y, theta = self.pose  
        xn = x + self.r * math.cos(theta)  
        yn = y + self.r * math.sin(theta)  
        elems += ax.plot([x,xn], [y,yn], color=self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color) 
        elems.append(ax.add_patch(c))
        result = self.wind()
        c = ax.quiver(self.pose[0], self.pose[1], result[0], result[1])
        elems.append(c)
        self.poses.append(self.pose)
    
    #風の情報を返す関数
    def wind(self):
        return self.wind_model.velocity_at_pos(self.pose[0], self.pose[1])
    
    #本体を進める関数
    @classmethod           
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        #回転速度の違いで分岐
        if math.fabs(omega) < 1e-10:
            return pose + np.array( [nu*math.cos(t0), 
                                     nu*math.sin(t0),
                                     omega ] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), 
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time ] )
    #状態更新する関数
    def one_step(self, time_interval, nu, omega):     
        self.pose = self.state_transition(nu, omega, time_interval, self.pose) 


# 群を成す粒子のクラス(ABC)

# In[45]:


class Particle_abc: 
    def __init__(self, init_pose, weight, direction, dis, option = False):
        self.pose = init_pose
        self.weight = weight
        self.dis = dis
        self.count = 0
        self.direction = direction
        self.option = option
        self.memo = []
        self.old = self.pose.copy()
        self.wind = np.array([0, 0]).T
    
    #風との角度を返す関数
    def angle(self, x, y):
        dot_xy = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        cos = dot_xy / (norm_x*norm_y)
        rad = np.arccos(cos)
        theta = rad * 180 / np.pi

        return theta
    
    #動作させる関数
    def motion_update(self, time, found, target, space):
        if found:
            j = np.random.randint(2)
            r = np.random.rand() * 2 - 1
            delta = (self.pose[j] - target.old[j])
            if delta == 0:
                wind = self.wind / np.linalg.norm(self.wind)
                delta = -wind[j]

            new = self.old.copy() 
            delta = r * delta
            if delta > self.dis:
                delta = self.dis
            a = np.zeros(2)
            a[j] = delta
            if self.option:
                theta = self.angle(a, self.direction)
                if theta < 90:
                    new[j] = self.old[j] - delta * time #近傍点計算
                else:
                    new[j] = self.old[j] + delta * time #近傍点計算
            else:
                theta = self.angle(a, self.wind)
                if theta < 45:
                        new[j] = self.old[j] - delta * time #近傍点計算 
                        
            if space.array_xy_region.contains(new[0], new[1]):
                self.pose = new        
            
            if self.weight > 0:
                self.count = 0
            else:
                self.count += 1

        
        else:
            p = self.pose + np.array([np.random.rand() * 6 - 3, np.random.rand() * 6 - 3]).T * time  
            if space.array_xy_region.contains(p[0], p[1]):
                    self.pose = p

# 群を成す粒子のクラス(PSO)
# In[46]:


class Particle_pso: 
    def __init__(self, init_pose, weight, dis, q = 0, option = False):
        self.pose = init_pose
        self.weight = weight
        self.per = init_pose
        self.dis = dis
        self.q = q
        self.option = option
        self.memo = []
        self.wind = np.array([0, 0]).T
        self.delta = np.array([0, 0]).T
    
    #風のとの角度を算出する関数
    def angle(self, x, y):

        dot_xy = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        cos = dot_xy / (norm_x*norm_y)
        rad = np.arccos(cos)
        theta = rad * 180 / np.pi

        return theta
    
    #動作させる関数
    def motion_update(self, time, glob, q_list, main_list, found, space, w = 0.5, b = 1.0, c = 1.0):
        if found == False:
            p = self.pose + np.array([np.random.rand() * 6 - 3, np.random.rand() * 6 - 3]).T * time  
            if space.array_xy_region.contains(p[0], p[1]):
                    self.pose = p
        else:
            self.delta = (w * self.delta) + (b * np.random.rand() * (self.per - self.pose)) +  (c * np.random.rand() * (glob - self.pose))
            if self.option:
                if self.q == 1:
                    repul = 0
                    r_core = 0.46
                    r_limit = 1
                    for i in range(len(q_list)):
                        diff = np.linalg.norm(q_list[i].pose - self.pose)
                        if diff > 1e-9:
                            if diff < r_core:
                                repul += self.q * q_list[i].q * (self.pose - q_list[i].pose) / (diff * r_core**2)
                            elif r_core < diff < r_limit:
                                repul += self.q * q_list[i].q * (self.pose - q_list[i].pose) / (diff**3)
                    self.delta += repul
                elif self.q == -1:
                    repul = 0
                    r_core = 3
                    for i in range(len(main_list)):
                        diff = np.linalg.norm(main_list[i].pose - self.pose)
                        if 1e-9< diff < r_core:
                            repul += (self.pose - main_list[i].pose) / (diff * r_core**2)
                    self.delta += repul
            if np.linalg.norm(self.delta) > self.dis:
                self.delta = (self.delta / np.linalg.norm(self.delta)) * self.dis
            self.delta[np.isnan(self.delta)] = 0
            if found == True:
                theta = self.angle(self.delta, self.wind)
                if theta > 45:
                    p = self.pose + self.delta * time
                    if space.array_xy_region.contains(p[0], p[1]):
                        self.pose = p
            else:
                p = self.pose - self.delta * time
                if space.array_xy_region.contains(p[0], p[1]):
                    self.pose = p
                else:
                    target = np.random.choice(main_list)
                    self.pose = target.pos +  np.array([np.random.rand(), np.random.rand()]).T


# 群全体のクラス

# In[47]:


class group:  
    def __init__(self, envmap,  sensor = None, abc = False):
        self.particles = []
        self.map = envmap
        self.sensor = sensor
        self.abc = abc
        self.pos = None
        self.found = False
        self.found_list = []
    
    #粒子を追加する関数
    def append_particles(self, init_pose, num, option = False):
        if not self.abc:
            self.popu_num = 1
            self.one = int(num / self.popu_num)
            self.q_num = int(self.one / 2)
        for i in range(num):
            dis = 0.5
            pose = init_pose + np.array([np.random.rand() * 2 + 2, (5 / num) * i]).T
            weight = -1e10
            if self.abc:
                direction = np.array([1,1])
                self.particles.append(Particle_abc(pose, weight, direction, dis, option = option))
            else:
                if (i % self.one) < self.q_num:
                    q = 1
                elif (i % self.one) == self.q_num:
                    q = -1
                else:
                    q = 0
                weight = self.sensor.data(pose)
                self.particles.append(Particle_pso(pose, weight, dis, q, option = option))
                
        if self.abc:
            self.pos  = pose.copy()
            self.best = self.sensor.data(self.pos)
        else:
            self.found = [False for i in range(self.popu_num)]
            self.pos = np.tile(pose,(self.popu_num,1))
            self.best = self.particles[:self.popu_num]
            self.main = self.particles[:self.popu_num]
        #最良の個体を決定する
        self.set_best()
        
    #最良の個体を設定する関数
    def set_best(self):
        if self.abc:
            i = np.argmax([p.weight for p in self.particles])
            if self.best < self.particles[i].weight:
                self.best = self.particles[i].weight
                self.pos = self.particles[i].pose.copy()
            if self.particles[i].weight > 0:
                self.found = True
            else:
                self.found = False
        else:
            for i in range(self.popu_num):
                j = np.argmax([p.weight for p in self.particles[i * self.one:(i + 1) * self.one]])
                best = self.sensor.data(self.pos[i])
                if best <= self.particles[(i * self.one) + j].weight:
                    self.best[i] = self.particles[(i * self.one) + j]
                    self.pos[i] = self.best[i].pose.copy()
                if self.particles[(i * self.one) + j].weight > 0:
                    self.found[i] = True
                else:
                    if self.found[i] == True:
                        self.found[i] = 2
                        popu_data = np.array([p.pose for p in self.particles[i * self.one:(i + 1) * self.one]])
                        x = np.mean(popu_data[:,0])
                        y = np.mean(popu_data[:,1])
                        self.pos[i] = np.array([x,y]).T
    
    #道中の計測値も更新に使用する関数
    def check(self, index):
        p = self.particles[index]
        sense = self.sensor.data(p.pose)
        if sense <= 0 and self.abc:
            if len(self.found_list) > 0:
                value_list = [np.linalg.norm(i.pose - p.pose) for i in self.found_list]
                index = np.argmin(value_list)
                dis = np.min(value_list)
                p.direction = p.old - self.found_list[index].pose
            else:
                dis = np.linalg.norm(self.pos - p.pose)
                p.direction = p.old - self.pos
            sense = -dis
        else:
            p.direction = p.wind.copy()
        
        p.memo.append(sense)
        if len(p.memo) > 10:
            record = np.mean(p.memo)
            p.memo = []
            p.memo.append(record)
    
    #更新関数
    def reset(self, index):
        
        p = self.particles[index]
        if self.abc:
            
            if len(p.memo) > 0:
                sense= np.mean(p.memo)
            else:
                sense = self.sensor.data(p.pose)
            
                if sense <= 0:
                    if len(self.found_list) > 0:
                        value_list = [np.linalg.norm(i.pose - p.pose) for i in self.found_list]
                        index = np.argmin(value_list)
                        dis = np.min(value_list)
                        p.direction = p.old - self.found_list[index].pose
                    else:
                        dis = np.linalg.norm(self.pos - p.pose)
                        p.direction = p.old - self.pos
                    sense = -dis
                else:
                    p.direction = p.wind.copy()
                
            if np.linalg.norm(p.pose - p.old) < 0.01:
                if sense > 0:
                    sense = self.sensor.data(p.pose)
                p.weight = sense
                p.old = p.pose.copy()  
            elif p.weight < sense:
                p.old = p.pose.copy()
                p.weight = sense
            else:
                p.pose = p.old.copy() 

        else:
            sense= self.sensor.data(p.pose)
            if p.weight <= sense:
                p.per = p.pose
            p.weight = sense
            
        p.memo = []
    
    #各粒子を移動させる関数
    def motion_update(self, time, goal):
        
        if self.abc:
            self.found_list = []
            for p in self.particles:
                if p.weight > 0:
                    self.found_list.append(p)
            if len(self.found_list) > 0:
                self.found = True
            else:
                self.found = False
        else: 
            count = 0
            current = count * self.one

        for i,p in enumerate(self.particles):
            if self.abc:
                k = np.random.choice(self.particles)
                p.motion_update(time, self.found, k, self.map)
                if p.count > 300:
                    p.pose = np.array([np.random.rand()*5, np.random.rand()*5]).T
                    p.old = p.pose.copy()
                    p.count = 0
            else:
                if (i % self.one) == 0:
                    if i != 0:
                        count += 1
                        current = count * self.one
                if self.found[count] == 2:
                    p.motion_update(time, self.pos[count], self.particles[current:current + self.q_num], self.main, self.found[count], self.map, c = 1e6)
                else:
                    p.motion_update(time, self.pos[count], self.particles[current:current + self.q_num], self.main, self.found[count], self.map)
                if p.q == -1:
                    self.main[count] = p

        self.set_best()
        
    #描画
    def draw(self, ax, elems):
        for p in self.particles:
            if self.abc:
                if p.weight > 0:
                    c = ax.scatter(p.pose[0], p.pose[1], s=45, marker=".", color="purple")
                else:
                    c = ax.scatter(p.pose[0], p.pose[1], s=45, marker=".", color="black")
            else:
                c = ax.scatter(p.pose[0], p.pose[1], s=45, marker=".", color="orange")
            elems.append(c)
            value_str = "%.2f" % (np.round(p.weight, 3))
            elems.append(ax.text(p.pose[0] + 0.1, p.pose[1], value_str, fontsize=15))
            
        if self.abc:
            c = ax.scatter(self.pos[0], self.pos[1], s=45, marker=".", color="blue")
            elems.append(c)
        else:
            for i in self.pos:
                c = ax.scatter(i[0], i[1], s=300, marker=".", color="blue")
                elems.append(c)


# ロボットの目標を定める関数

# In[48]:


class Agent(): 
    def __init__(self, pose, time_interval, estimator):
        self.nu = 0
        self.omega = 0
        self.pose = pose
        self.estimator = estimator
        self.robots = []
        self.goal = []
        self.time_interval = time_interval
        self.win = False
        self.time = 0
        self.prev_nu = 0
        self.prev_omega = 0
      
    #ロボットを追加する関数
    def append_robots(self, num, wind_model):
        self.goal = np.zeros(num)
        for i in range(num):
            #初期値設定
            posi = self.estimator.particles[i].pose
            pose = np.array([posi[0], posi[1], 0]).T
            if not self.estimator.abc:
                if self.estimator.particles[i].q == 1:
                    self.robots.append(Robot(pose, wind_model, color="red"))
                elif self.estimator.particles[i].q == -1:
                    self.robots.append(Robot(pose, wind_model, color="blue"))
                else:
                    self.robots.append(Robot(pose, wind_model, color="black"))
            else :
                self.robots.append(Robot(pose, wind_model, color="blue"))

            
    
    #目標に向かう関数
    @classmethod   
    def policy(cls, pose, goal):
        x, y, theta = pose
        dx, dy = goal[0] - x, goal[1] - y
        direction = int((math.atan2(dy, dx) - theta)*180/math.pi)   #ゴールの方角（degreeに直す）
        direction = (direction + 360*1000 + 180)%360 - 180      #方角を-180〜180[deg]に正規化（適当。ロボットが-1000回転すると破綻）
        
        if (direction < 20) and (direction > -20): nu, omega = 1.0, 0.0
        elif direction > 135:  nu, omega = -1, 0.0
        elif direction < -135: nu, omega = -1, 0.0
        elif direction > 0: nu, omega = 0.0, 2.0
        else: nu, omega = 0.0, -2.0
            
        return nu, omega
    
    #速度決定関数
    def one_step(self, observation=None):
        
        if self.win:
            return
        
        for i,r in enumerate(self.robots):
            if np.linalg.norm(r.pose[:2] - self.estimator.particles[i].pose) < r.r:
                if self.goal[i] == 0:
                    self.estimator.reset(i)
                    self.goal[i] = 1
                self.estimator.particles[i].wind = r.wind()
        
        if np.sum(self.goal) == len(self.robots):
            self.estimator.motion_update(self.time_interval, self.goal)
            self.goal[:] = 0
        for i,r in enumerate(self.robots):
            nu, omega = self.policy(r.pose, self.estimator.particles[i].pose)
            if np.linalg.norm(r.pose[:2] - self.estimator.particles[i].pose) > r.r:
                r.one_step(self.time_interval, nu, omega)
                self.estimator.check(i)
            if self.estimator.particles[i].weight > 0.5:
                if self.estimator.map.check(r):
                    self.win = True
   
    #描画
    def draw(self, ax, elems): 
        self.estimator.draw(ax, elems)
        for r in self.robots:
            r.draw(ax, elems)
        #x, y, t = self.estimator.pos 


# 障害物を管理するクラス

# In[49]:


class Map(object):

    def __init__(self, array_xy_region, array_z, n_x, n_y, puff_mol_amount,
                 kernel_rad_mult=3):
        self.odors = []
        self.array_xy_region = array_xy_region
        self.array_z = array_z
        self.n_x = n_x
        self.n_y = n_y
        self.dx = array_xy_region.w / n_x  # calculate x grid point spacing
        self.dy = array_xy_region.h / n_y  # calculate y grid point spacing
        # precompute constant used to scale Gaussian kernel amplitude
        self._ampl_const = puff_mol_amount / (8*np.pi**3)**0.5
        self.kernel_rad_mult = kernel_rad_mult

    def _puff_kernel(self, shift_x, shift_y, z_offset, r_sq, even_w, even_h):
        # kernel is truncated to min +/- kernel_rad_mult * effective puff
        # radius from centre i.e. Gaussian kernel with >= kernel_rad_mult *
        # standard deviation span
        # (effective puff radius is (r_sq - (z_offset/k_r_mult)**2)**0.5 to
        # account for the cross sections of puffs with centres out of the
        # array plane being 'smaller')
        # the truncation will introduce some errors
        shape = (2 * (r_sq * self.kernel_rad_mult**2 - z_offset**2)**0.5 /
                 np.array([self.dx, self.dy]))
        # depending on whether centre is on grid points or grid centres
        # kernel dimensions will need to be forced to odd/even respectively
        shape[0] = self._round_up_to_next_even_or_odd(shape[0], even_w)
        shape[1] = self._round_up_to_next_even_or_odd(shape[1], even_h)
        # generate x and y grids with required shape
        [x_grid, y_grid] = 0.5 + np.mgrid[-shape[0] // 2:shape[0] // 2,
                                          -shape[1] // 2:shape[1] // 2]
        # apply shifts to correct for offset of true centre from nearest
        # grid-point / centre
        x_grid = x_grid * self.dx + shift_x
        y_grid = y_grid * self.dy + shift_y
        # compute square radial field
        r_sq_grid = x_grid**2 + y_grid**2 + z_offset**2
        # output scaled Gaussian kernel
        return (self._ampl_const  * np.exp(-r_sq_grid / (r_sq**2)))
    
    def _puff_conc_dist(self, x, y, z, px, py, pz, r_sq):
        return (
            self._ampl_const  *
            np.exp(-((x - px)**2 + (y - py)**2 + (z - pz)**2) / (r_sq**2))
        )

    def calc_conc_point(self, puff_array, x, y, z=0):
        # filter for non-nan puff entries and separate properties for
        # convenience
        px, py, pz, r_sq = puff_array[~np.isnan(puff_array[:, 0]), :].T
        return self._puff_conc_dist(x, y, z, px, py, pz, r_sq).sum(-1)
    
    def func(self, x, y, z=0):
        total = 0 
        for lm in self.odors:
            total += self.calc_conc_point(lm.puff_array, x, y)
        return total
            
    @staticmethod
    def _round_up_to_next_even_or_odd(value, to_even):
        value = math.ceil(value)
        if to_even:
            if value % 2 == 1:
                value += 1
        else:
            if value % 2 == 0:
                value += 1
        return value

    def generate_single_array(self, puff_array):
        # initialise concentration array
        conc_array = np.zeros((self.n_x, self.n_y))
        # loop through all the puffs
        for (puff_x, puff_y, puff_z, puff_r_sq) in puff_array:
            # to begin with check this a real puff and not a placeholder nan
            # entry as puff arrays may have been pre-allocated with nan
            # at a fixed size for efficiency and as the number of puffs
            # existing at any time interval is variable some entries in the
            # array will be unallocated, placeholder entries should be
            # contiguous (i.e. all entries after the first placeholder will
            # also be placeholders) therefore break out of loop completely
            # if one is encountered
            if np.isnan(puff_x):
                break
            # check puff centre is within region array is being calculated
            # over otherwise skip
            if not self.array_xy_region.contains(puff_x, puff_y):
                continue
            # finally check that puff z-coordinate is within
            # kernel_rad_mult*r_sq of array evaluation height otherwise skip
            puff_z_offset = (self.array_z - puff_z)
            if abs(puff_z_offset) / puff_r_sq**0.5 > self.kernel_rad_mult:
                continue
            # calculate (float) row index corresponding to puff x coord
            p = (puff_x - self.array_xy_region.x_min) / self.dx
            # calculate (float) column index corresponding to puff y coord
            q = (puff_y - self.array_xy_region.y_min) / self.dy
            # calculate nearest integer or half-integer row index to p
            u = math.floor(2 * p + 0.5) / 2
            # calculate nearest integer or half-integer row index to q
            v = math.floor(2 * q + 0.5) / 2
            # generate puff kernel array of appropriate scale and taking
            # into account true centre offset from nearest half-grid
            # points (u,v)
            kernel = self._puff_kernel(
                (p - u) * self.dx, (q - v) * self.dy, puff_z_offset, puff_r_sq,
                u % 1 == 0, v % 1 == 0)
            # compute row and column slices for source kernel array and
            # destination concentration array taking in to the account
            # the possibility of the kernel being partly outside the
            # extents of the destination array
            (w, h) = kernel.shape
            r_rng_arr = slice(int(max(0, u - w / 2.)),
                              int(max(min(u + w / 2., self.n_x), 0)))
            c_rng_arr = slice(int(max(0, v - h / 2.)),
                              int(max(min(v + h / 2., self.n_y), 0)))
            r_rng_knl = slice(int(max(0, -u + w / 2.)),
                              int(min(-u + w / 2. + self.n_x, w)))
            c_rng_knl = slice(int(max(0, -v + h / 2.)),
                              int(min(-v + h / 2. + self.n_y, h)))
            # add puff kernel values to concentration field array
            conc_array[r_rng_arr, c_rng_arr] += kernel[r_rng_knl, c_rng_knl]
        return conc_array

    def append_odor(self, odor):       # ランドマークを追加
        odor.id = len(self.odors)           # 追加するランドマークにIDを与える
        self.odors.append(odor)
        
    def check(self, robo):
        robo_pos = np.array(robo.pose[:2])
        for i,lm in enumerate(self.odors):
            sorce = np.array(lm._new_puff_params[:2])
            if np.linalg.norm(robo_pos - sorce) < robo.r:
                del self.odors[i]
                if len(self.odors) == 0:
                    return True
        return False
    
    def one_step(self, time_interval):
        for lm in self.odors: lm.update(time_interval)
    
    def draw(self, ax, elems):                 # 描画
        conc_array = np.zeros((self.n_x, self.n_y))
        conc_im = ax.imshow(conc_array.T, extent=self.array_xy_region, vmin=0., vmax=1e5, cmap='Reds', alpha = 0.5)
        for lm in self.odors:
            conc_array += np.fliplr(self.generate_single_array(lm.puff_array))
            c = ax.scatter(lm._new_puff_params[0], lm._new_puff_params[1], s=100, marker=".", color="green")
            elems.append(c)
        
        conc_im.set_data(conc_array.T)
        elems += [conc_im]


# センサーのクラス

# In[50]:


class Sensor:
    def __init__(self, env_map, border=0.3):
        self.map = env_map
        self.lastdata = []
        self.border = (env_map._ampl_const / 1e3)
        self.error = (env_map._ampl_const / 1e4)
    
    #見えたデータを作成する関数
    def data(self, sens_pose):

        mol = self.map.func(sens_pose[0],sens_pose[1])
        if mol < self.border:
            mol = 0
        else:
            mol += (np.random.rand() * 2 - 1)* self.error
        
        self.lastdata = mol 
        return (mol / self.map._ampl_const)


# In[51]:


if __name__ == '__main__':   ###name_indent
    
    #パラメータの設定
    seed = 20180517
    rng = np.random.RandomState(seed)
    
    time_interval = 0.1
    area_max = 5.
    area_min = 0.
    
    wind_model_params = {
    'u_av': 0.5,  # Mean x-component of wind velocity (u).
    'v_av': 0.,  # Mean y-component of wind velocity (v).
    'k_x': 1.,  # Diffusivity constant in x direction.
    'k_y': 1.,  # Diffusivity constant in y direction.
    'noise_gain': 3.,  # Input gain constant for boundary condition noise generation.
    'noise_damp': 1,  # Damping ratio for boundary condition noise generation.
    'noise_bandwidth': 1,  # Bandwidth for boundary condition noise generation.
    }
    wind_model_params['use_original_noise_updates'] = True

    wind_model_params['sim_region'] = Rectangle(x_min=area_min, y_min=area_min, x_max=area_max, y_max=area_max)
    wind_model_params['n_x'] = 2 * int(area_max)  # Number of grid points in x-direction.
    wind_model_params['n_y'] = 2 * int(area_max)  # Number of grid points in y-direction.

    plume_model_params = {
    'centre_rel_diff_scale': 0.5,  # Scaling coefficient of diffusion of puffs
    'puff_release_rate': 20,  # Initial radius of the puffs
    'puff_init_rad': 0.20,  # Initial puff radius
    'puff_spread_rate': 0.025  # Rate of puff size spread over time
    }
    plume_model_params['sim_region'] = Rectangle(x_min=area_min, x_max=area_max, y_min=area_min, y_max=area_max)
    plume_model_params['source_pos'] = (1.0, 6, 0.)
    plume_model_params.update({
        'init_num_puffs': 10,
        'max_num_puffs': 200,
        'model_z_disp': True,
    })
    
    array_gen_params = {
    'array_z': 0.,  # Height on z-axis at which to calculate concentrations
    'n_x': 100,  # Number of grid points to sample at across x-dimension.
    'n_y': 100,  # Number of grid points to sample at across y-dimension.
    'puff_mol_amount': 8.3e8  # Molecular content of each puff
    }
    
    q_pso = 0
    d_abc = 0
    pso = 0
    abc = 0
    
    q_pso_time = []
    d_abc_time = []
    pso_time = []
    abc_time = []
    
    #実験の実行
    for l in range(1):
        #実験環境の作成
        world = World(200, time_interval, area_min, area_max,debug=True)

        # Run wind model forward to equilibrate
        wind_model = WindModel(rng=rng, **wind_model_params)

        ### 地図を生成して3つランドマークを追加 ###
        m1 = Map(array_xy_region=Rectangle(x_min=area_min, x_max=area_max, y_min=area_min, y_max=area_max), **array_gen_params)
        odors_num = 1
        for i in range(odors_num):
            plume_model_params['source_pos'] = (1., (i * 3) + 2.5, 0.)
            m1.append_odor(PlumeModel(rng=rng, wind_model=wind_model, **plume_model_params))

        m2 = Map(array_xy_region=Rectangle(x_min=area_min, x_max=area_max, y_min=area_min, y_max=area_max), **array_gen_params)
        odors_num = 1
        for i in range(odors_num):
            plume_model_params['source_pos'] = (1., (i * 3) + 2.5, 0.)
            m2.append_odor(PlumeModel(rng=rng, wind_model=wind_model, **plume_model_params))
            
        m3 = Map(array_xy_region=Rectangle(x_min=area_min, x_max=area_max, y_min=area_min, y_max=area_max), **array_gen_params)
        odors_num = 1
        for i in range(odors_num):
            plume_model_params['source_pos'] = (1., (i * 3) + 2.5, 0.)
            m3.append_odor(PlumeModel(rng=rng, wind_model=wind_model, **plume_model_params))
            
        m4 = Map(array_xy_region=Rectangle(x_min=area_min, x_max=area_max, y_min=area_min, y_max=area_max), **array_gen_params)
        odors_num = 1
        for i in range(odors_num):
            plume_model_params['source_pos'] = (1., (i * 3) + 2.5, 0.)
            m4.append_odor(PlumeModel(rng=rng, wind_model=wind_model, **plume_model_params))
        
        for k in range(100):
            wind_model.one_step(time_interval)
            #m.one_step(time_interval)
        world.append(wind_model)    
        world.append(m1)
        world.append(m2)
        world.append(m3)
        world.append(m4)

        ##水たまりの追加##
        #world.append(Puddle((20, -5), (30, 5), 0.5))

        ### ロボットを作る ###
        popu = 10
        initial_pose = np.array([0, 0]).T
                
        estimator1 = group(m1, sensor = Sensor(m1), abc = True)
        estimator1.append_particles(initial_pose, popu, option = True)
        a1 = Agent(initial_pose, time_interval, estimator1)
        a1.append_robots(popu, wind_model = wind_model)
        world.append(a1)
    
        estimator2 = group(m2, sensor = Sensor(m2), abc = False)
        estimator2.append_particles(initial_pose, popu, option = True)
        a2 = Agent(initial_pose, time_interval, estimator2)
        a2.append_robots(popu, wind_model = wind_model)
        world.append(a2)

        estimator3 = group(m3, sensor = Sensor(m3), abc = True)
        estimator3.append_particles(initial_pose, popu, option = False)
        a3 = Agent(initial_pose, time_interval, estimator3)
        a3.append_robots(popu, wind_model = wind_model)
        world.append(a3)
        
        estimator4 = group(m4, sensor = Sensor(m4), abc = False)
        estimator4.append_particles(initial_pose, popu, option = False)
        a4 = Agent(initial_pose, time_interval, estimator4)
        a4.append_robots(popu, wind_model = wind_model)
        world.append(a4)

        ### アニメーション実行 ###
        world.draw()
        

        if a1.win:
            d_abc += 1
            print("d_abc")
            d_abc_time.append(a1.time)
        else:
            d_abc_time.append(200)
        
        if a2.win:
            q_pso += 1
            print("q_pso")
            q_pso_time.append(a2.time)
        else:
            q_pso_time.append(200)
        
        if a3.win:
            abc += 1
            print("abc")
            abc_time.append(a3.time)
        else:
            abc_time.append(200)
        
        if a4.win:
            pso += 1
            print("pso")
            pso_time.append(a4.time)
        else:
            pso_time.append(200)
        print(l)
        
        del world, wind_model, m1, m2, m3, m4, estimator1, estimator2, estimator3, estimator4, a1, a2, a3, a4,
    
    #実験結果出力
    d_abc_time = np.array(d_abc_time)
    q_pso_time = np.array(q_pso_time)    
    abc_time = np.array(abc_time)
    pso_time = np.array(pso_time)
    print("d_abc")
    print(d_abc)
    print(d_abc_time)
    print(np.round(np.mean(d_abc_time), 3))
    print(np.round(np.std(d_abc_time), 3))
    print("  ")
    print("q_pso")
    print(q_pso)
    print(q_pso_time)
    print(np.round(np.mean(q_pso_time),3))
    print(np.round(np.std(q_pso_time),3))
    print("  ")
    print("abc")    
    print(abc)
    print(abc_time)
    print(np.round(np.mean(abc_time), 3))
    print(np.round(np.std(abc_time), 3))
    print("  ")
    print("pso")
    print(pso)
    print(pso_time)
    print(np.round(np.mean(pso_time),3))
    print(np.round(np.std(pso_time),3))
    


# In[ ]:




