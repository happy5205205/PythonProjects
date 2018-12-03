import matplotlib.pyplot as plt
import seaborn as sns


def plot_type1(pokemon_data):
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    #Type_1的数量统计图
    sns.countplot(x='Type_1', data=pokemon_data)
    plt.title('主要类别的数量统计')
    plt.xticks(rotation='vertical')
    plt.xlabel('主要类别')
    plt.ylabel('数量')

    #Type_2的数量统计图
    plt.subplot(1, 2, 2, sharey=ax1)
    sns.countplot(x='Type_2', data=pokemon_data)
    plt.title('副类别的数量统计')
    plt.xticks(rotation='vertical')
    plt.xlabel('副类别')
    plt.ylabel('数量')
    plt.tight_layout()
    plt.savefig('./主副类数量统计.png')
    plt.show()


def plot_Egg_Group(pokemon_data):
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    # Egg_Group_1的数量统计图
    sns.countplot(x='Egg_Group_1', data=pokemon_data)
    plt.title('蛋群分组1的数量统计')
    plt.xticks(rotation=60)
    plt.xlabel('蛋群分组1')
    plt.ylabel('数量')

    # Egg_Group_2的数量统计图
    plt.subplot(1, 2, 2, sharey=ax1)
    sns.countplot(x='Egg_Group_2', data=pokemon_data)
    plt.title('蛋群分组2的数量统计')
    plt.xticks(rotation='vertical')
    plt.xlabel('蛋群分组2')
    plt.ylabel('数量')
    plt.tight_layout()
    plt.savefig('./蛋群分组的数量统计.png')
    plt.show()


def plot_other(pokemon_data):
    plt.figure(figsize=(12, 6))
    # Color的数量统计图
    ax1 = plt.subplot(2, 3, 1)
    sns.countplot(x='Color', data=pokemon_data)
    plt.title('颜色的数量统计')
    plt.xticks(rotation=60)
    plt.xlabel('颜色')
    plt.ylabel('数量')

    # isLegendary 的数量统计图
    ax1 = plt.subplot(2, 3, 2, sharey=ax1)
    sns.countplot(x='isLegendary', data=pokemon_data)
    plt.title('是否为传说类型的数量统计')
    plt.xlabel('是否为“传说”')
    plt.ylabel('数量')

    # hasGender 的数量统计图
    plt.subplot(2, 3, 3, sharey=ax1)
    sns.countplot(x='hasGender', data=pokemon_data)
    plt.title('是否有性别的数量统计')
    plt.xlabel('是否有性别')
    plt.ylabel('数量')

    # hasMegaEvolution 的数量统计图
    plt.subplot(2, 3, 4, sharey=ax1)
    sns.countplot(x='hasMegaEvolution', data=pokemon_data)
    plt.title('是否有Mega进化的数量统计')
    plt.xlabel('是否有Mega进化')
    plt.ylabel('数量')

    # 身形 的数量统计图
    plt.subplot(2, 3, 5, sharey=ax1)
    sns.countplot(x='Body_Style', data=pokemon_data)
    plt.xticks(rotation=90)
    plt.title('身形的数量统计')
    plt.xlabel('身形')
    plt.ylabel('数量')

    # 第n代 的数量统计图
    plt.subplot(2, 3, 6, sharey=ax1)
    sns.countplot(x='Generation', data=pokemon_data)
    plt.title('第n代的数量统计')
    plt.xlabel('第n代')
    plt.ylabel('数量')

    plt.tight_layout()
    plt.savefig('./其它变量数量.jpg')
    plt.show()


def plot_count_num(pokemon_data):
    # 数值型的统计分布
    plt.figure(figsize=(12, 6))
    plt.title('数值型的统计分布')
    numeric_cols = ['Total', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'Pr_Male',
                    'Height_m', 'Weight_kg', 'Catch_Rate']
    for i in range(len(numeric_cols)):
        plt.subplot(3, 4, i+1)
        sns.distplot(pokemon_data[numeric_cols[i]].dropna())
        plt.xlabel(numeric_cols[i])
    plt.tight_layout()
    plt.savefig('./数字型.jpg')
    plt.show()