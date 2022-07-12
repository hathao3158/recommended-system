#from distutils.sysconfig import customize_compiler
#from tkinter import Variable
import streamlit as st
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import pickle
#import re

#********************************************************************************************************************************
# SOURCE CODE
#********************************************************************************************************************************

content=pd.read_csv('content.csv')
content_short=pd.read_csv('content_short.csv')
similar_product=pd.read_csv('similar_product.csv')
similar_product.columns.values[0] = "Key"
similar_product['similar_products']=similar_product[["Product_1","Product_2","Product_3","Product_4","Product_5"]].values.tolist()

review=pd.read_csv('ReviewRaw.csv')
rating=content[['item_id','1','2','3','4','5']]
for col in rating.columns[1:]:
    rating[col]=rating[col].fillna(0).astype(int)
rating_detail=pd.melt(rating, id_vars='item_id',value_vars=['1','2','3','4','5'])
rating_df=rating_detail.pivot(index='variable',values='value',columns='item_id').reset_index()
rating_df=rating_df.sort_values(by='variable',ascending=False)
rating_df['rating']=rating_df['variable'].apply(lambda x: str(x)+"⭐")

customized=pd.read_csv('customized_product.csv')
customized['customized_products']=customized[["Product_1","Product_2","Product_3","Product_4","Product_5"]].values.tolist()
customer_list=review['customer_id'].unique().tolist()

#********************************************************************************************************************************
# FUNCTION
#********************************************************************************************************************************

# PART 2 - CONTENT RECOMMEND SYSTEM
def key_product_info(product_id):
    product_index=content[content['item_id']==product_id].index[0]
    st.write("Product Name:",content[content['item_id']==product_id]['name'].values[0],"-Product ID:",product_id)
    # print("Product Info:")
    # print(content.iloc[product_index].light_description)
    st.write("Product Link:","https://tiki.vn/"+content.iloc[product_index].url[8:])
    st.write("Product Image:",content.iloc[product_index].image)
def similar_product_info(product_id):
    st.write("Product Name:",content[content['item_id']==product_id]['name'].values[0],"-Product ID:",product_id)
#------------------------------------------------------------------------------------------------------------------------------
# PART 3 - CONTENT RECOMMENDATION APPLICATION

def product_info_layout(product_id):
    product_df=content[content['item_id']==product_id]
    name=product_df['name'].values[0]
    key_group=product_df['key_group'].values[0]
    sub_group_1=product_df['sub_group_1'].values[0]
    brand=product_df['brand'].values[0]
    image=product_df['image'].values[0]
    price=product_df['price'].values[0]
    list_price=product_df['list_price'].values[0]
    discount=str(round((-list_price+price)/list_price*100,0))+"%"
    
    col1, col2= st.columns([4,3])
    with col1:
        st.image(image,use_column_width='always')
    with col2:
        st.subheader(name)
        st.write(f'{brand}\t|\t{key_group}\t|\t{sub_group_1}')
        st.metric(label="Original Price", value=list_price)
        st.metric(label="Discounted Price", value=price,delta=discount)
def extend_info(value):
    TTCT_idx=value.index("THÔNG TIN CHI TIẾT")
    MTSP_idx=value.index("MÔ TẢ SẢN PHẨM")
    for text in value[:MTSP_idx]:
        if text=="THÔNG TIN CHI TIẾT":
            st.write(f'#### **{text}**')
        else:
            st.write(f"- {text}")
def extend_description(value):
    MTSP_idx=value.index("MÔ TẢ SẢN PHẨM")
    for text in value[MTSP_idx:-1]:
        if text=="MÔ TẢ SẢN PHẨM":
            st.write(f'#### **{text}**')
        else:
            st.write(f"- {text}")

def product_description(product_id):
    st.subheader("Product Information")
    product_df=content[content['item_id']==product_id]
    link="https://tiki.vn/"+product_df['url'].values[0][8:]
    description=product_df['light_description'].values[0]
    value=description.split("\n")
    TTCT_idx=value.index("THÔNG TIN CHI TIẾT")
    MTSP_idx=value.index("MÔ TẢ SẢN PHẨM")
    extend_info(value)
    extend_description(value)
    st.write("Giá sản phẩm trên Tiki đã bao gồm thuế theo luật hiện hành. Tuy nhiên tuỳ vào từng loại sản phẩm hoặc phương thức địa chỉ giao hàng mà có thể phát sinh thêm chi phí khác như phí vận chuyển phụ phí hàng cồng kềnh ...")
    st.write(f'Reference Link: {link}')

def product_review_1(product_id):
    st.subheader("Product Review")
    col1, col2= st.columns([4,5])
    with col1:
        total_rating=str(int(content[content['item_id']==product_id]['total_rating']))
        ave_rating_score=str(content[content['item_id']==product_id]['average_rating_score'].tolist()[0])
        string="     "+ave_rating_score+"⭐"
        st.write(f"""
        \n
        ##### Average Rating Scores:
        """)
        st.write(f'''
        ## {string}
        ''')
        st.write(f"""
        ({total_rating} ratings from customers.)
        """)
    with col2:
        st.write(f'##### No. of reviews by rating:\n')
        fig=  plt.figure(figsize=(5,2.5))
        g=sns.barplot(data=rating_df[['variable',product_id]], x=product_id, y='variable',color='dodgerblue')
        plt.xlabel("")
        plt.ylabel("")
        g.tick_params(bottom=False)
        st.pyplot(fig)

def product_review_2(product_id):
    my_expander=st.expander("Detail Product Review from Tiki customers")
    with my_expander:
        if product_id in review['product_id'].unique().tolist():
            #st.write("##### Reviews with detail content:")
            review['content']=review['content'].fillna("")
            review['rating_new']=review['rating'].apply(lambda x: str(x)+"⭐")
            product_review=review[(review['product_id']==product_id)&(review['content']!="")].sort_values('rating',ascending=False).reset_index()
            product_review=product_review[['name','rating_new','title','content']]
            st.table(product_review)
        else:
            st.write('This product have no reviews.') 

def similar_product_rec_1(i):
    similar_list=similar_product[similar_product['Key']==item_id]['similar_products'].values[0]
    product_df=content[content['item_id']==similar_list[i]]
    st.image(product_df['image'].values[0])
    st.write(product_df['name'].values[0])

def similar_product_rec_2(i):
    similar_list=similar_product[similar_product['Key']==item_id]['similar_products'].values[0]
    product_df=content[content['item_id']==similar_list[i]]
    st.metric(label="Discounted Price",value=product_df['price'].values[0])
    my_expander=st.expander("Product details")
    with my_expander:
        product_description(similar_list[i])
    
def similar_product_rec(product_id):
    col1,col2,col3,col4,col5=st.columns([2,2,2,2,2])
    with col1:
        similar_product_rec_1(0) 
    with col2:
        similar_product_rec_1(1)     
    with col3:
        similar_product_rec_1(2)
    with col4:
        similar_product_rec_1(3)
    with col5:
        similar_product_rec_1(4)
    col1,col2,col3,col4,col5=st.columns([2,2,2,2,2])
    with col1:
        similar_product_rec_2(0) 
    with col2:
        similar_product_rec_2(1)     
    with col3:
        similar_product_rec_2(2)
    with col4:
        similar_product_rec_2(3)
    with col5:
        similar_product_rec_2(4)

#------------------------------------------------------------------------------------------------------------------------------
# PART 5 - COLLABORATIVE RECOMMENDATION APPLICATION
def top_customize_product(cus_id):
    product_id_list=customized[customized['customer_id']==cus_id]['customized_products'].values[0]
    for i in range(0, len(product_id_list)):
        product_id_list[i] = int(product_id_list[i])
    for pro in product_id_list:
        product_info_layout(pro)
        my_expander=st.expander("Product details")
        with my_expander:
            product_description(pro)
            product_review_1(pro)
        st.write("-"*100)

def customized_product(cus_id):
    if cus_id in customer_list:
        cus_name=review[review['customer_id']==cus_id]['name'].values[0]
        st.write(f'##### Hi {cus_name},')
        st.write('##### Tiki would like to recommend you:\n')
        st.write(top_customize_product(cus_id))
    else:
        st.write(f'##### Hi beloved user,')
        st.write('##### Tiki cannot find your ID  on Tiki Platform.')
        st.write('##### Please find our top products with highest numbers of recommendations in below:\n')
        for pro in [299461,1600005,47321729,405243,8141868]:
            product_info_layout(pro)
            my_expander=st.expander("Product details")
            with my_expander:
                product_description(pro)
                product_review_1(pro)
            st.write("-"*100)  

#********************************************************************************************************************************
# GUI
#********************************************************************************************************************************

st.title('Tiki Recommend System')
#st.markdown(f'<p style="color:#00a3cc;font-size:42px;border-radius:2%;">{title}</p>', unsafe_allow_html=True)#background-color:#00a3cc;
st.title(" ")

menu = ['Bussiness Objective','Content Recommend System','Content Recommendation Application','Collaborative Recommend System','Collaborative Recommendation Application']
choice = st.sidebar.selectbox('Menu',menu)

#------------------------------------------------------------------------------------------------------------------------------
# PART 1 - BUSINESS OBJECTIVE
if choice == 'Bussiness Objective':
    
    st.subheader('Bussiness Objective/Problem')
    st.image('logo-tiki.png')
    st.write('''
    ##### Tiki is an "all in one" commercial ecosystem, in there is tiki.vn, which is a standing e-commerce website Top 2 in Vietnam, top 6 in Southeast Asia.

On tiki.vn website, many advanced support utilities have been deployed high user experience and they want to build many more conveniences.

Assuming Tiki has not implemented Recommender System and you are required to implement this system. What you will do?
    ''')
    
    st.subheader(' ')
    st.subheader('Solution Recommendation')
    st.write('''
    ##### Based on the above requirements, we need to build Recommendation System on tiki.vn to give product suggestions to users/customers.

There are 2 types of model we can use for tiki.vn:
- Content based filtering
- Collaborative filtering
    ''')
    st.image('recommend_system.png')

#------------------------------------------------------------------------------------------------------------------------------
# PART 2 - CONTENT RECOMMEND SYSTEM
elif choice == 'Content Recommend System':
    st.subheader('Content Recommend System')
    st.write('''
    We will build Content Recommend System base on Product Information such as Product Name, Product Category & Product Description. 
    Content Recommend System Application:
    - If customers choose a specific product, Content Recommend System will suggest 5 similar products.

    ''')
    st.write('''
    ### 1. Data Understanding:
    ''')
    st.write('##### Read Product Raw Data')
    st.dataframe(content_short[['item_id', 'name', 'description', 'rating', 'price', 'list_price',
       'brand', 'group', 'url', 'image']].head())
    st.write(f'We can see that:')
    st.write('1. Column "group" has long names but they can be classified into key group & sub groups.')
    st.write('2. To give recommendation base on content, we can merge column "name" & column "description" together under new column "content", which will support buildding our Content-based Recommend System.')
    
    st.write('''
    ### 2. Data Preparation:
    ''')
    st.write('##### New Product DataFrame after cleaning:')
    st.dataframe(content_short[['item_id', 'name', 'url', 'image', 'description', 'rating', 'price','list_price', 'brand', 'group', 'key_group', 'sub_group_1','sub_group_2', 'sub_group_3', 'content']].head())
    key_group=content['key_group'].unique()
    key_group_1=', '.join(key_group)
    st.write('##### Review New Product DataFrame:')
    st.write(f'There are {len(key_group)} key groups:\n {key_group_1}')
    product_by_group=content[['key_group','item_id']].groupby('key_group').count().sort_values('item_id',ascending=False).reset_index()
    #st.table(product_by_group)
    fig1 =  plt.figure(figsize=(10,4))
    sns.barplot(data=product_by_group, y='item_id', x='key_group',color='mediumseagreen')
    plt.xlabel("Key Group")
    plt.ylabel("Numbers of products")
    plt.xticks(rotation=80)
    plt.title('Numbers of products by key group')
    st.pyplot(fig1)
    st.write('''
    ### 3. Content-Based Recommend System Model:
    ''')
    st.write("""
    We use 2 models for content-based recommend system: Gensim & Cosine Similarity.
    Both Gensim & Cosine Similarity show good results for content-based recommendation.
    However, we would like to use Cosine Similarity model for our content-based recommend system.
    ##### MODEL RESULT - Top 5 similar products for each specific product:
    Below is a result dataframe with:
    - "Key" is an itemID of a specific product
    - 5 next columns are 5 itemID of 5 products which are similar to a specific product from index
    """)

    st.dataframe(similar_product.head())
    st.write("""
    ##### END-USERS VIEW
    However, when we apply to end users, we need to translate item_id to product name.\n
    **With key item_id: 48102821:**
    """)
  
    key_product_info(48102821)
    st.write("""
    **Content-based recommend system will suggest 5 similar products:**
    """)
    for i in range(0, len(similar_product['similar_products'][0])):
        similar_product_info(int(similar_product['similar_products'][0][i]))

#------------------------------------------------------------------------------------------------------------------------------
# PART 3 - CONTENT RECOMMENDATION APPLICATION
elif choice == 'Content Recommendation Application':
    st.header("Product Recommendation with Content Recommend System")
    product_list = content['name'].values
    selected_product = st.selectbox("Type or select a product from the dropdown",product_list)
    if st.button('🔍 Search'):
        item_id=content[content['name']==selected_product]['item_id'].values[0]
        st.subheader("-"*79)
        ## Product Key Metrics
        product_info_layout(item_id)
        ## Product Description
        product_description(item_id)
        ## Product Review
        st.subheader("-"*79)
        product_review_1(item_id)
        product_review_2(item_id)
        ## Product Recommendation
        st.subheader("-"*79)
        st.subheader("Top 5 Similar Products")
        similar_product_rec(item_id)
        st.subheader("-"*79)
#------------------------------------------------------------------------------------------------------------------------------
# PART 4 - COLLABORATIVE RECOMMEND SYSTEM
elif choice == 'Collaborative Recommend System':
    st.subheader('Collaborative Recommend System')
    st.write('''
    We will build Collaborative Recommend System base on CustomerID, ProductID & rating. 
    Collaborative Recommend System Application:
    - For each customer, Collaborative Recommend System will suggest 5 customized products base on their historical rating of other products.

    ''')
    st.write('''
    ### 1. Data Understanding:
    ''')
    st.write('##### Read Product Raw Data')
    st.dataframe(review.head())
    st.write(f'''
    We only need 3 columns for Collaborative Recommend System "customer_id","product_id","rating".
    Therefore, drop other columns''')
        
    st.write('''
    ### 2. Data Preparation:
    ''')
    st.write('##### New Product DataFrame after cleaning:')
    st.dataframe(review[["customer_id","product_id","rating"]].head())
    
    st.write('''
    ### 3. Content-Based Recommend System Model:
    ''')
    st.write("""
    We use 2 models for collaborative recommend system: ALS & SurPRISE.
    Within SurPRISE, we test with many models such as SVD(), SVDpp(), NMF(), SlopeOne(), BaselineOnly(),KNNBasic(), KNNBaseline(), KNNWithMeans(), CoClustering(), KNNWithZScore(),CoClustering(), KNNWithZScore().
    After try different models, we would like to use SurPRISE with BaselineOnly model for our collaborative recommend system.
    ##### MODEL RESULT - Top 5 similar products for each specific product:
    Below is a result dataframe with:
    - customerID 
    - 5 next columns are 5 itemID of 5 products which are recommened to customer with that customerID
    """)

    st.dataframe(customized[['customer_id','Product_1','Product_2','Product_3','Product_4','Product_5','customized_products']].head())
    st.write("""
    ##### END-USERS VIEW
    However, when we apply to end users, we need to translate customer_id to customer name & item_id to product name.\n
    With customer_id: 709310, Collaborative Recommend System would like to suggest in template below:
    """)
  
    cus_name=review[review['customer_id']==709310]['name'].values[0]
    st.write(f'##### Hi {cus_name},')
    st.write('##### Tiki would like to recommend you:\n')
    for i in range(0, len(customized['customized_products'][0])):
        similar_product_info(int(customized['customized_products'][0][i]))

#------------------------------------------------------------------------------------------------------------------------------
# PART 5 - COLLABORATIVE RECOMMENDATION APPLICATION
elif choice == 'Collaborative Recommendation Application':
    st.subheader("-"*79)
    st.header("Product Recommendation with Collaborative Recommend System")
    st.image('tiki-banner3.png')
    st.subheader("-"*79)
    st.subheader('Best Selling Categories at Tiki')
    col1,col2,col3,col4,col5=st.columns([2,2,2,2,2])
    with col1:
        st.image('thiet_bi_phu_kien_so.jpg')
    with col2:
        st.image('dien_tu_dien_lanh.jpg')
    with col3:
        st.image('may_anh_quay_phim.jpg')
    with col4:
        st.image('may_tinh_laptop.jpg')
    with col5:
        st.image('dien_thoai_may_tinh_bang.jpg')
    
    col1,col2,col3,col4,col5=st.columns([2,2,2,2,2])
    with col1:
        st.write('Thiết Bị Số - Phụ Kiện Số')
    with col2:
        st.write('Điện Tử - Điện Lạnh')
    with col3:
        st.write('Máy Ảnh - Máy Quay Phim')
    with col4:
        st.write('Laptop - Máy Vi Tính - Linh kiện')
    with col5:
        st.write('Điện Thoại - Máy Tính Bảng')
    st.subheader("-"*79)
    st.subheader('## Find out special recommendations for you today:')
    max_id=max(customized['customer_id'])
    min_id=min(customized['customer_id'])
    input_id = st_tags(label='Enter your Customer_id:',text='Press → then Enter',
                        suggestions= [str(x) for x in review['customer_id'].to_list()],
                        maxtags = 1,key='1')
    if len(input_id)>0:
        cus_id = float(input_id[0])
        customized_product(cus_id)
        st.write('Note: Remember to clear current user_id before input a new one')
    st.subheader("-"*79)
