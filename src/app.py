from TownScapeENV import *
import streamlit as st
from stable_baselines3 import PPO
import base64
from pathlib import Path
from PIL import ImageDraw

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def gif_base64(frames, fps):
    imgs = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    buf = BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


st.title("This is an Urban Planning Assistant")
shape_opt_val = ((5,5),(6,8),(8,8),(10,10),(12,16))
shape_opt_str = [f"{i}x{j}" for i,j in shape_opt_val]

st.header("Settings")
shape_inp = st.radio("What size of the city do you want?",shape_opt_str, horizontal = True,index = 3)
shape = shape_inp.split('x')
shape = [eval(ele) for ele in shape]

map = np.zeros(shape, dtype = np.uint8)
area = shape[0]*shape[1]

home_inp = int(st.number_input("How many houses do you want", min_value = 0, max_value = area//4))
work_inp = int(st.number_input("How many workplaces do you want", min_value = 0, max_value = area//4))

obstacle_opt = ('Empty','Random','Custom')
obstacle_inp = st.selectbox('Obstacle type',obstacle_opt)
if obstacle_inp == 'Empty': obsc_map = []
elif obstacle_inp== 'Random': obsc_map = None
else:
    defualt_s = ''
    for i in range(shape[0]):
        defualt_s += np.array2string(np.zeros(shape[1], dtype= np.uint8)).strip('][')
        defualt_s += '\n'
    defualt_s = defualt_s[:-1] 
    
    s_inp = st.text_area('Obstacle Set', defualt_s, key='txt')
    obsc_map = []
    for row in s_inp.split('\n'):
        temp = row.split(' ')
        temp = [eval(ele) for ele in temp]
        obsc_map.append(temp)
    obsc_map = np.array(obsc_map, dtype = np.uint8)
    st.text('current obstacle')
    st.text(obsc_map)

shape = tuple(shape)
env0 = townScape(shape)
this_requirements = {'map': obsc_map, 'home':home_inp, 'target': work_inp}
obs, _ = env0.reset(requirements = this_requirements)

st.header("Current Requirements")
st.markdown(f"Size of the city: {shape}")
st.markdown(f"Number of House(s): {env0.rem_home}")
st.markdown(f"Number of Workplace(s): {env0.rem_target}")

st.header('Program')
bta_sta = st.button('Proceed')
if bta_sta: 
    model_nam = f'{shape[0]}x{shape[1]}_200' # CHANGE model size HERE
    model = PPO.load(MODEL_DIR / model_nam, custom_objects={
        "observation_space": env0.observation_space,
        "action_space": env0.action_space,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    })

    this_requirements['map'] = env0.get_compressed_obs()
    this_requirements['home'] = env0.rem_home
    this_requirements['target'] = env0.rem_target

    img_width = 320


    st.header("Initial State")
    _,mid,_ = st.columns([1,3,1])   
    init_img = env0.render()
    with mid:
        st.image(init_img, 'Initial State', width = img_width)

    env_list = []
    bar = st.progress(0,'Loading...')

    eps = 50
    for i in range(1,eps+1):
        bar.progress(i*int(100/eps),'Loading...')
        env = townScape(shape)
        state, _ = env.reset(requirements = this_requirements)
        done = False
        while not done:
            action, _ = model.predict(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        tf,suc_c,suc_r,_, sim_info = env.simulate()
        if sim_info == 'phase1' or sim_info == 'blank': md = 0
        elif len(tf) == 0: md = 0
        elif sim_info == 'square': md = 0
        else: md = np.median(np.array(tf,dtype = np.double))
        env_list.append({'env': env, 'tf':tf, 'md':md,'sc':suc_c,'sr':suc_r})
    bar.progress(100, 'Finshed')
    env_list = sorted(env_list, key = lambda ele: ele['md'], reverse=True)


    n_rank = 3
    st.header(f"Selected {n_rank} city(s)")
    for i in range(n_rank):
        with st.expander(f"Rank{i+1}"):
            ele = env_list[i]
            img = ele['env'].render()
            tf = ele['tf']
            md = ele['md']
            suc_c = ele['sc']
            suc_r = ele['sr']
            st.subheader('Result')
            with st.container():
                _,mid,_ = st.columns([1,2,1]) 
                with mid:
                    st.image(img, 'Result', width = img_width)
                    st.text(f"Median of traffic_flow = {md}")
                    st.text(f"Success = {suc_c} ({suc_r*100}%)")
            st.subheader("Simulate")
            with st.container():
                _,_,_,sim_frames,_ = ele['env'].simulate(render = True)
                fps = 2
                if not isinstance(sim_frames, list) or len(sim_frames) == 0 or not isinstance(tf, list):
                    st.info("No car can reach a workplace in this plan, so there is nothing to simulate.")
                    continue
                simUrl = gif_base64(list(sim_frames), fps)

                #Description GIF
                txtImgs = []
                for i in range(len(tf)):
                    img = Image.new('RGB', (img_width, 40), (14, 17, 23))
                    d = ImageDraw.Draw(img)
                    text = f"time = {i+1}, traffic flow = {tf[i]}"
                    d.text((15, 15), text, align = 'center', fill=(226, 228, 230))
                    txtImgs.append(np.array(img))
                txtUrl = gif_base64(txtImgs, fps)
                _,mid,_ = st.columns([1,2,1]) 
                with mid:
                    st.markdown(f'<img src="data:image/gif;base64,{simUrl}" alt="sim gif" width="{img_width}">',unsafe_allow_html=True)
                    st.markdown(f'<img src="data:image/gif;base64,{txtUrl}" alt="sim gif" width="{img_width}">',unsafe_allow_html=True)
                

                
 
