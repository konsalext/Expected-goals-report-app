#importing libraries

import streamlit as st
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from highlight_text import HighlightText
from mplsoccer import VerticalPitch , Pitch
from PIL import Image
import random



st.set_page_config(layout="wide")
st.header("The Expected Goals app :soccer: :chart_with_upwards_trend:")

tab_one , tab_two , tab_three = st.tabs([":question: How to use the app",":soccer: Get the report !",":bar_chart: More info ..."])
@st.cache_resource
def log_reg():
    log_reg_model = joblib.load("the_model.joblib")
    return log_reg_model

xG_model=log_reg()

@st.cache_data
def sample_one():
    df_city = pd.read_csv("City_vs_arsenal_events.csv")
    return df_city
@st.cache_data
def sample_two():
    df_osfp=pd.read_csv("OSFP_vs_Genk_events.csv")
    return df_osfp

first_sample_game=sample_one()
second_sample_game=sample_two()

data_for_City = first_sample_game.to_csv().encode("utf-8")
data_for_osfp = second_sample_game.to_csv().encode("utf-8")


def get_xG_probabilities (df):
    # we clean the events and teams column because if there is extra white space during typing ,  df.loc won't work afterwards
    df["events_cleaned"]=df.Event.str.strip()
    df["teams_cleaned"]=df.Team.str.strip()
    
#then we change coordinates to meters , find shot distance from center of the goal , if it was a header or not
    df["x_for_oppo"]=100-df["X"]
    df["y_for_oppo"]=abs(100-df["Y"])
    scale_x_wy=105/100
    scale_y_wy=68/100
    df["x_meters"]=df["X"]*scale_x_wy
    df["y_meters"]=df["Y"]*scale_y_wy
    df["distance_coords"]=np.sqrt((df["X"]-100)**2 + (df["Y"]-50)**2)
    df["distance_meters"]=np.sqrt((df["x_meters"]-105)**2 + (df["y_meters"]-34)**2)
    df["header_or_not"]=df.events_cleaned.apply(lambda x:1 if x=="Header" else 0)
    
#then calculate the angle (see Wikipedia's law of cosines page)

    df["x_side_dist"]=np.sqrt((df["x_meters"]-105)**2 + (df["y_meters"]-37.66)**2)
    df["y_side_dist"]=np.sqrt((df["x_meters"]-105)**2 + (df["y_meters"]-30.34)**2)
    df["first_part"]=df["x_side_dist"]**2 + df["y_side_dist"]**2 - 7.32**2
    df["second_part"]=2*df["x_side_dist"]*df["y_side_dist"]
    df["third_part"]=df["first_part"] / df['second_part']
    df["angle_rads"]=np.arccos(df["third_part"])
    df["angle_degrees"]= df["angle_rads"]*(180/np.pi)

# build the numpy array to feed into the model
    X_mtrs = df["x_meters"].values
    distance_from_goal = df["distance_meters"].values
    the_angle = df["angle_degrees"].values
    head_or_no = df["header_or_not"].values
    
    per_shot_list =[[i,k,l,m] for i , k, l , m in zip(X_mtrs , distance_from_goal,the_angle,head_or_no)]
    per_shot_array = np.array(per_shot_list)

# we take the model predictions and put them in a list called xG_prob
    xG_prob=[]
    for p in range(0,len(per_shot_array)):
        single_val=np.array(per_shot_array[p],ndmin=2)
        xg_single=xG_model.predict_proba(single_val).reshape(1,-1)
        xG_prob.append(xg_single[0][1])

# for intepretability we round the expected goal values to 3 decimal points and add them to a new lists
    xG_prob_rounded=[]
    
    for q in range(0,len(xG_prob)) :
        rounded_value = round(xG_prob[q] , 3)
        xG_prob_rounded.append(rounded_value)
    
# we append a new column containing the xG values to the dataframe
    df["xGoals"] = xG_prob_rounded
    
# rename the events column
    df = df.rename({"events_cleaned":"Events","teams_cleaned":"Teams"},axis=1)
    
# for ease of use we round angle and distance to three decimal points    
    dist_list = df["distance_meters"].values
    angle_list = df["angle_degrees"].values
    dist_rounded =[]
    angle_rounded=[]
    for v in range(0 , len(dist_list)) :
        rounded_distance_val = round(dist_list[v] , 3)
        dist_rounded.append(rounded_distance_val)
    for w in range(0 , len(angle_list)):
        rounded_angle_val = round(angle_list[w] , 3)
        angle_rounded.append(rounded_angle_val)
        
    df["goal_mouth_angle"] = angle_rounded
    df["distance_from_goal"] = dist_rounded
    
    final_df=df[["Teams","Player","Events","X","Y","x_for_oppo",
                 "y_for_oppo","x_meters","distance_coords","distance_from_goal","goal_mouth_angle","xGoals"]]
    final_df=final_df.sort_values(by="Teams",ascending=True).reset_index(drop=True)
    return final_df

with tab_one:
    fcpython_instructions = Image.open("fc_python_img.png")
    csv_instructions = Image.open("instructions_for_csv.png")
    the_other_side = Image.open("logging_on_opposite_side.png")
    tutorial_url="https://youtu.be/j96xcb2cUlg"
    st.snow()
    st.markdown("_Hello and welcome !_ Intended for football professionals (coaches , analysts etc.) purpose of this app\
                is to provide the user with the famous expected goals metric.")
    st.markdown("Say _-as an analyst-_ you code your team's game and you log all the goal chances for both teams.\
                Having watched the footage now you want a bit more objective information about the **quality** of those shots ,\
                _did we actually create high-quality chances while limiting those of the opponent ?_ In this case , having\
                an xG metric makes all the difference.")
    st.markdown("Problem one is that it costs money to get these metrics from data providers (Opta, Statsbomb, Wyscout etc.)\
                and some lower-level teams **may not have the budget** for that.Problem two is simply that the aforementioned\
                providers don't cover all the leagues of a country , for example to my knowledge there is **no data available for Greek \
                National C Division teams** (Γ' Εθνική)")
    subcol_seven,subcol_eight = st.columns(2)
    with subcol_seven :
        st.markdown("Enter the app ! The process is quite simple it's all done in three steps : First you load the events file , then you enter\
                team names and scorelines and _voila_ report is ready in a few seconds !")
        st.markdown("You can watch this video-tutorial on the right on learning how to use the app properly , or _(if you don't speak Greek)_ you can\
                    read the instructions below.")
    with subcol_eight :
          st.video(tutorial_url)
   
    subcol_nine , subcol_ten = st.columns(2)
    with subcol_nine :
        st.markdown("The file type accepted is CSV(comma-separated values) , and for the tagging/coding process\
                    you can use [FcPython's online tagging tool](https://fcpythonvideocoder.netlify.app/)\
                    as you can see the process is quite straightforward _(just make sure you log all chances to the right goal , for both\
                    teams)_ :")
        st.markdown("Just make sure that every event you see in the footage (for both teams) occuring in the left goal , you're logging it on\
                **exactly the opposite side of the pitch** like in the picture below (opposite in both x and y axis) , and make sure you are doing that for both teams'\
                events _(a technicality with the app , it understands only the right goal as attacking third)_.Whatever happens on the right goal you log it as usual.")
    with subcol_ten:
        st.image(fcpython_instructions , caption = "the instructions step-by-step")
        st.image(the_other_side , caption="How to log events happening on the left goal")
    st.markdown('''Using FcPython for the coding is not necessary , all you need is a CSV-type file with 4 columns named **_"Teams , Event,\
                 X, Y"_** and populated as per the instructions below.Be very careful with the syntax , it **has to be spot-on** as \
                given in the instructions or else the app will fail !''')
    st.image(csv_instructions , caption="how to format the csv file")
    st.markdown("So , let's get started shall we ?")
    st.subheader("Step 1 : select the file you want to upload")
    the_uploaded_data_file = st.file_uploader("Upload csv file here :")
    if the_uploaded_data_file is not None :
        the_data = pd.read_csv(the_uploaded_data_file)
        edited_df = get_xG_probabilities(the_data)
        time.sleep(2)
        st.success("File uploaded successfully !", icon ="✅")
    with st.expander ("In case you don't have data and you want to try it anyway... 👇"):
        st.markdown("..feel free to use one of the two games i have tagged for \
        you :\n - Man.City vs Arsenal 1-1 (Community shield 23-24)\n - Olympiacos vs Genk 1-0\
                (Europa League qualifiers 23-24)")
        city_button = st.download_button(label="City-Arsenal 1-1",data=data_for_City ,
                               file_name=f"match events for City vs Arsenal.csv",
                               mime="text/csv")
        osfp_button = st.download_button(label="Olympiacos - Genk 1-0",data=data_for_osfp,file_name="match events for Olympiacos vs Genk.csv",
                                     mime="text/csv")

with tab_two :

    col_one , col_two = st.columns(2)
    with col_one :
        st.subheader("Step 2 : Team names")
        st.markdown("_(please make sure the team names' syntax is **exactly** the same as the one in the csv file you are going to upload , or else the app won't work)_")
        home_team = st.text_input ("the name of the home team and Enter 👇")
        away_team = st.text_input("the name of the away team and Enter 👇 ")
        home_team_clean=home_team.strip()
        away_team_clean=away_team.strip()
        home_goals = st.selectbox("How many goals did the home team score ?",[0,1,2,3,4,5,6,7,8,9,10],0)
        away_goals = st.selectbox("How many goals did the away team score ?",[0,1,2,3,4,5,6,7,8,9,10],0)
        home_goals = int(home_goals)
        away_goals = int(away_goals)
        

        if home_team and away_team and the_uploaded_data_file is not None :
            

            #separate dataframes to home and away teams based on user's input
            df_home = edited_df.loc[edited_df["Teams"]==home_team_clean]
            df_away=edited_df.loc[edited_df["Teams"]==away_team_clean]

            #separate shots from headers (they're gonna be plotted differently)
            df_only_shots_home = df_home.loc[df_home["Events"]=="Shot"]
            df_only_shots_away=df_away.loc[df_away["Events"]=="Shot"]
            df_only_headers_home=df_home.loc[df_home["Events"]=="Header"]
            df_only_headers_away=df_away.loc[df_away["Events"]=="Header"]
                    
            # define all the metrics into variables
            xgoals_home = df_home["xGoals"].tolist()
            xgoals_away = df_away["xGoals"].tolist()
            chances_home = df_home.shape[0]
            chances_away = df_away.shape[0]
            headers_home = df_home[df_home["Events"]=="Header"].shape[0]
            headers_away = df_away[df_away["Events"]=="Header"].shape[0]
            total_xg_home = sum(xgoals_home)
            total_xg_away= sum(xgoals_away)
            total_xg_home = round(total_xg_home , 3)
            total_xg_away = round(total_xg_away , 3)
            xg_per_shot_home= total_xg_home / len(xgoals_home)
            xg_per_shot_away= total_xg_away / len(xgoals_away)
            xg_per_shot_home=round(xg_per_shot_home,3)
            xg_per_shot_away=round(xg_per_shot_away,3)
            avg_dist_home = df_home["distance_from_goal"].mean()
            avg_dist_away=df_away["distance_from_goal"].mean()
            avg_dist_home=round(avg_dist_home,2)
            avg_dist_away=round(avg_dist_away,2)
            dist_coords_home=df_home["distance_coords"].mean()
            dist_coords_away=df_away["distance_coords"].mean()
            dist_coords_home=round(dist_coords_home,2)
            dist_coords_away=round(dist_coords_away,2)
            home_dist_line= 0 + dist_coords_home
            away_dist_line=100 - dist_coords_away
            # get the top players in terms of xg
            df_home_grouped=df_home.groupby("Player").sum()
            df_away_grouped=df_away.groupby("Player").sum()
            df_home_grouped = df_home_grouped.sort_values(by="xGoals",ascending = False)
            df_away_grouped = df_away_grouped.sort_values(by="xGoals" , ascending = False)
            top_home_player = df_home_grouped.index[0]
            top_away_player = df_away_grouped.index[0]
            top_home_xg = df_home_grouped["xGoals"][0]
            top_away_xg=df_away_grouped["xGoals"][0]
            top_home_xg = round(top_home_xg , 3)
            top_away_xg = round(top_away_xg , 3)

            # xPoints simulator (over 10,000 games)
            home_wins=0
            away_wins=0
            draws=0
            n=10000

            for t in range(n):
                home_team_goals=0
                away_team_goals=0
                for j in range(len(xgoals_home)):
                    x=random.random()
                    if x < xgoals_home[j]:
                        home_team_goals +=1
                for k in range(len(xgoals_away)):
                    y=random.random()
                    if y < xgoals_away[k]:
                        away_team_goals +=1
                if home_team_goals > away_team_goals :
                    home_wins +=1
                elif away_team_goals > home_team_goals :
                    away_wins +=1
                else :
                    draws +=1


            # define xPoints metrics into variables        
            home_points = home_wins*3 + draws
            away_points = away_wins*3 + draws
            xPoints_home=home_points/n
            xPoints_away=away_points/n
            xPoints_home=round(xPoints_home,2)
            xPoints_away=round(xPoints_away,2)
            home_win_perc=home_wins/n*100
            away_win_perc=away_wins/n*100
            home_win_perc=round(home_win_perc,2)
            away_win_perc=round(away_win_perc,2)

            
    with col_two :
        st.subheader("Step 3 : Results time !")
        
        if home_team and away_team and the_uploaded_data_file is not None :
            data_for_export_df=edited_df[["Teams","Player","Events","distance_from_goal","goal_mouth_angle","xGoals"]]
            data_for_export = data-for_export_df.to_csv().encode("utf-8")
            data_button = st.download_button(label="Download the data :bar_chart:",data=data_for_export ,
                                   file_name=f"data for {home_team} vs {away_team}.csv",
                                   mime="text/csv")
            if st.button("generate xGoals match report"):
                
                pitch = Pitch(pitch_type="wyscout",axis=False,label=False,tick=False,half=False,pitch_color="#323332",line_color="white",
                     line_alpha=0.65,linewidth=2,goal_type="box",corner_arcs=True,line_zorder=1)
                props_labels = dict(boxstyle='round', facecolor='beige',edgecolor="white", alpha=0.3)
                props_num_home =dict(boxstyle='round', facecolor='firebrick',edgecolor="white", alpha=0.55)
                props_num_away=dict(boxstyle='round', facecolor='dodgerblue',edgecolor="white", alpha=0.55)

                fig ,axs = pitch.grid (figheight=6,grid_height=0.82 , title_height=0.04,endnote_height=0.05,bottom=None,left=None,axis=False)
                fig.set_facecolor("#323332")
        

                # plot title
                the_title= HighlightText(x=0.20,y=0.50,s=f"xGoals report : <{home_team_clean}> vs <{away_team_clean}> ",
                                         highlight_textprops=[{"color":"firebrick","fontsize":21,"fontweight":"bold"}
                                                              ,{"color":"dodgerblue","fontsize":21,"fontweight":"bold"}],
                                         fontsize=16,color="white",fontname="monospace",ax=axs["title"])
                # xgoals circles
                home_xgoals_from_shots=pitch.scatter(df_only_shots_home.x_for_oppo,df_only_shots_home.y_for_oppo,s=df_only_shots_home["xGoals"]*900,
                                          c="firebrick",edgecolors="white", alpha=0.7,marker="o",ax=axs["pitch"])
                away_xgoals_from_shots=pitch.scatter(df_only_shots_away.X,df_only_shots_away.Y,s=df_only_shots_away["xGoals"]*900,
                                         c="dodgerblue",edgecolors="white", marker="o",alpha=0.7 ,ax=axs["pitch"])
                if df_only_headers_home.shape[0] !=0:
                    home_headed_chances = pitch.scatter(df_only_headers_home.x_for_oppo , df_only_headers_home.y_for_oppo,s=df_only_headers_home["xGoals"]*900 ,
                                                        c="firebrick" , hatch="///",edgecolor="white",marker="o",alpha=0.7,ax=axs["pitch"])
                if df_only_headers_away.shape[0] !=0:
                    away_headed_chances = pitch.scatter(df_only_headers_away.X,df_only_headers_away.Y , s=df_only_headers_away["xGoals"]*900 , c="dodgerblue",
                                                        hatch="///",edgecolor="white",marker="o",alpha=0.7,ax=axs["pitch"])

                # plot average distance of shots
                dist_home_annot=pitch.lines(0,2.5,home_dist_line,2.5,color="firebrick",alpha=0.60,comet=True,lw=9,
                                            transparent=True,ax=axs["pitch"])
                first_ball=pitch.scatter(home_dist_line+1.3, 2.5, marker="football",ax=axs["pitch"],s=260)

                dist_away_annot=pitch.lines(100,2.5,away_dist_line,2.5,color="dodgerblue",alpha=0.60,comet=True,lw=9,
                                            transparent=True,ax=axs["pitch"])
                second_ball=pitch.scatter(away_dist_line-1.3, 2.5, marker="football",ax=axs["pitch"],s=260)

                home_dist_annot=axs["pitch"].text(3,9.5,s=f"avg.shot distance {avg_dist_home} m.",fontsize=9,fontfamily="monospace",
                                              bbox=props_num_home,color="white",zorder=2)
                away_dist_annot=axs["pitch"].text(72,9.5,s=f"avg.shot distance {avg_dist_away} m.",fontsize=9,fontfamily="monospace",
                                              bbox=props_num_away,color="white",zorder=2)
                
                # the annotations for the metrics (goals,xg,xp,win%,chances,headers)
                goals_label=axs["pitch"].text(46.9,11,s="Goals",fontsize=10,fontfamily="monospace",
                                              bbox=props_labels,color="white",zorder=2)
                xg_label=axs["pitch"].text(46.4,21,s="Total xG",fontsize=10,fontfamily="monospace",
                                              bbox=props_labels,color="white",zorder=2)
                xgshot_label=axs["pitch"].text(46.4,31,s="xG / shot",fontsize=10,fontfamily="monospace",
                                              bbox=props_labels,color="white",zorder=2)
                xp_label=axs["pitch"].text(46.9,41,s="xPoints",fontsize=10,fontfamily="monospace",
                                              bbox=props_labels,color="white",zorder=2)
                winprob_label=axs["pitch"].text(46,51,s="Win prob.%",fontsize=10,fontfamily="monospace",
                                              bbox=props_labels,color="white",zorder=2)
                chances_label=axs["pitch"].text(44.5,61,s="Goal attempts",fontsize=10,fontfamily="monospace",
                                              bbox=props_labels,color="white",zorder=2)
                headed_label=axs["pitch"].text(43,71,s="Headed attempts",fontsize=10,fontfamily="monospace",
                                              bbox=props_labels,color="white",zorder=2)
                #the annotation of the actual numbers
                home_goals_num=axs["pitch"].text(42.7,11,home_goals,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_home)
                away_goals_num=axs["pitch"].text(56,11,away_goals,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_away)

                home_xg_num=axs["pitch"].text(37,21,total_xg_home,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_home)
                away_xg_num=axs["pitch"].text(58,21,total_xg_away,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_away)

                home_xgshot_num=axs["pitch"].text(37,31,xg_per_shot_home,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_home)
                away_xgshot_num=axs["pitch"].text(59,31,xg_per_shot_away,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_away)

                home_xp_num=axs["pitch"].text(39,41,xPoints_home,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_home)
                away_xp_num=axs["pitch"].text(58,41,xPoints_away,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_away)

                home_prob_num=axs["pitch"].text(38,51,home_win_perc,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_home)
                away_prob_num=axs["pitch"].text(60.2,51,away_win_perc,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_away)

                home_chances_num=axs["pitch"].text(39,61,chances_home,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_home)
                away_chances_num=axs["pitch"].text(62,61,chances_away,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_away)

                home_head_num=axs["pitch"].text(38,71,headers_home,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_home)
                away_head_num=axs["pitch"].text(63,71,headers_away,fontsize=10,fontfamily="monospace",fontweight="bold",
                                                color="white",zorder=2,bbox=props_num_away)

                player_with_most_xg_home = axs["pitch"].text(1.5,98,s=f"{top_home_player} ({top_home_xg} xG)",bbox=props_num_home,
                                                             fontsize=9,fontfamily="monospace" , color="white")
                player_with_most_xg_away = axs["pitch"].text(51.5,98,s=f"{top_away_player} ({top_away_xg} xG)",bbox=props_num_away,
                                                             fontsize=9,fontfamily="monospace" , color="white")

                

                the_caption = axs["endnote"].text(0.40,0.26,\
                s="* bigger circle = higher-xG chance\n* xPoints and win probability simmed over 10.000 games\n* by Alexiou Kon/nos",\
                color="white",fontstyle="italic",fontweight="light",fontsize=9,fontfamily="monospace",ha="left")

                st.success("report generated successfully !",icon="✅")
                if home_team=="Olympiacos" and away_team=="Genk" :
                    st.info("i noticed that you used one of the pre-tagged games i provided for you.Just for reference and comparison , Wyscout model's xG values for this one were\
                            Olympiacos (1,29 xG) and Genk (1,40 xG) so a difference of 0,11 xG in favor of Genk.Does the report you get resemble these results ?",icon="ℹ️")
                if home_team=="City" and away_team=="Arsenal":
                    st.info("i noticed that you used one of the pre-tagged games i provided for you.Just for reference and comparison , Wyscout model's xG values for this one were\
                            City (0,94 xG) and Arsenal (1,04 xG) so a difference of 0,10 xG in favor of Arsenal.Does the report you get resemble these results ?",icon="ℹ️")
                plt.savefig("the_plot.png",dpi=200,bbox_inches="tight")
                st.pyplot(fig)
                with open("the_plot.png","rb") as file :
                    plot_buttnn= st.download_button(label="Download match report :bar_chart:",data=file,
                                               file_name=f"{home_team} vs {away_team} match report.png",
                                                mime="image/png")

with tab_three :
    feature_corr_img = Image.open("the_grouped_df.png")
    the_flowchart_img=Image.open("the_flowchart.jpg")
    auc_curve_plot = Image.open("roc_auc_curve_final.png")
    col_three , col_four = st.columns(2)
    with col_three:
        st.subheader("The data")
        st.markdown("**Wyscout** and **Statsbomb** open data:")
        st.markdown(" - from Wyscout : men's top-5 European leagues 2017/18 + World Cup 2018 + EURO 2016.")
        st.markdown(" - from Statsbomb : top-5 European leagues 2015/16 + World Cup 2022 + India Super league 2021/22 + EURO 2020.\n - a combined\
        dataset of 83.371 goal attempts , 69.478 shots and 13.893 headers.After removing null values we got a final tally of 83365.") 
        st.subheader("Feature and model selection")
        st.markdown('''Commercial xG models routinely use a lot of additional features _(such as if the shot assist was a cross or no , \
        if there is defensive pressure etc.)_ to get more accurate predictions , here we went with **four** basic ones : firstly the **distance from \
        goal** in meters along the horizontal axis.Secondly the **distance from the center of the goal** in meters.The third feature was the **goal-mouth angle**\
        aka _"how much of the goal does the attacker sees"_ in degrees , and finally **if the chances was a header or not** (binary values , 0 and 1).And as it turns out these\
        four alone give a strong enough signal : below is the dataset grouped according to whether the chances ended in a goal (1) or not (0).The chances ending\
        with a goal are taken much closer to the opponent's goal (shown both by x_meters and distance) and with a bigger goal-mouth angle compared with the chances\
        that a goal did not come.Which makes me feel better about the machine learning model i selected , ''')
        st.image(feature_corr_img,caption="the relationships between the features and the taget variable")
        st.markdown(", since **logistic regression** _-known mostly for its straightforward nature-_ performs better when the independent variables are\
                    correlated with the target variable.Logistic regression also performs better when fewer features are used , which is our case exactly.")
        st.markdown(" - train-test split was 81/19 ,\n - L2 regularization used for avoiding overfitting.")
        st.subheader("Evaluating model performance")
        subcol_one ,subcol_two , subcol_three , subcol_four , subcol_five = st.columns(5)
        with subcol_one:
            st.metric("Accuracy","0.9011")
        with subcol_two:
            st.metric("cross validation","0.9011")
        with subcol_three:
            st.metric("Precision" ,"0.653")
        with subcol_four:
            st.metric("Recall" , "0.050")
        with subcol_five:
            st.metric("AUC-score" , "0.7856")
                            
        st.markdown(" Model accuracy is about 90/100 shown also by the 6-fold cross validation performed which is quite good.While the model is kinda terrible at\
                    predicting whether a chance is actually going to be a goal _(recall score)_ the ones that actually get classified as goals are almost always goals \
                    _(precision)_ .\n - _(regarding the low recall score , i think the scarcity of goals in general in the sport -it is the nature of football- and\
                    the subsequent imbalance of 0s and 1s in the target variable is messing with those results.)_")
        st.markdown('''Anyway , after all the purpose of an xGoals model is not to give us definite predictions like _"Goal or not , 0 or 1"_ but to provide \
                    **goal probabilities** for each chance , and in that aspect _(as shown in the ROC-AUC curve on the right)_ the model does pretty well , way better\
                    than just the "eye test" , thus giving us an objective metric of evaluating chances.''')
        
    with col_four:
        st.image(the_flowchart_img , caption = "The whole process from start to finish")
        st.image(auc_curve_plot , caption = "the AUC score is pretty good")
                    
    with st.expander("Are there any limitations to this model ? 🤔"):
        st.markdown('''Yes there are :\n - The dataset used to train the model is considered **rather small**.For a sufficiently accurate logistic regression model we need a **minimum of 50.000\
                    instances** in the training set , and with **83.365** we are well above that.Nevertheless those kinds of models only get more accurate the more data you feed\
                    into them.\n - The model evaluates chances only from 11x11 men's football matches and it **does not apply** to any youth competitiions , where the dimensions of the\
                    pitch and/or the goal might be idfferent.\n - Furthermore , the data used (top-5 leagues , World Cup , EURO) kinda _"tilts+_ towards the top level of the game ,\
                    and might not be as representative as it could be of how the game is played in the lower divisions.''')
    with st.expander("How about some extra info on xG ? :soccer: :bar_chart:"):
        st.markdown (" - the probability of your average chance being a goal is about **10%**. In this dataset , for a total of 83.371 chances we had 8.442 goals (9,87%).\
                        \n - The xG of a penalty kick is **0.76** _(by the way this is a non-penalty xG model so please don't log any penalties , just use the 0.76 constant instead)_.\
                        \n - according to a lot of providers such as OPTA , an xG value of about **0.3** and higher qualifies as a big chance.")
    with st.expander("About the author"):
        st.markdown(" - My name is Konstantinos Alexiou and i live in Athens , Greece.\n - I have a Bsc. in Physical Education and Sports Sciences as well as\
                    a Master's in performance analysis in\
                    football.\n - In October  of 2023 i will be starting my Msc. in Data Science and Machine Learning , thought this project was a good warm-up in that direction.\
                    \n - You can connect with me on [LinkedIn](www.linkedin.com/in/konstantinos-alexiou-40013796).\n - Hope you like the app , cheers 😊😊")
            
                    
            


                    

        
        
        
                 
    
        
    
