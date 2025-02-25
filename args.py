import argparse


# def get_parser():
#     parser = argparse.ArgumentParser() 
#     parser.add_argument("--model", type=str, default="gpt-4")
#     parser.add_argument("--eps", type=float, default=3.0)
#     parser.add_argument("--eps_fx", type=float, default=1.0)
#     # parser.add_argument("--eps_fx", type=str, default='noeps')
#     parser.add_argument("--K", type=int, default=1500)
#     parser.add_argument("--text", type=str, default='text')
#     parser.add_argument("--task", type=str, default='cnn_daily_1000')
#     parser.add_argument("--raw_path", type=str, default='output/cnn_daily_1000/cnn_daily_1000_processed.csv')
#     return parser


#imdb
# def get_parser():
#     parser = argparse.ArgumentParser() 
#     parser.add_argument("--model", type=str, default="gpt-4")
#     parser.add_argument("--eps", type=float, default=2.0)
#     parser.add_argument("--eps_fx", type=float, default=2.0)
#     # parser.add_argument("--eps_fx", type=str, default='noeps')
#     parser.add_argument("--K", type=int, default=20)
#     parser.add_argument("--text", type=str, default='text')
#     parser.add_argument("--task", type=str, default='imdb')
#     parser.add_argument("--raw_path", type=str, default='output/imdb/data.csv')
#     return parser

#20_news
# def get_parser():
#     parser = argparse.ArgumentParser() 
#     parser.add_argument("--model", type=str, default="gpt-4")
#     parser.add_argument("--eps", type=float, default=2.0)
#     parser.add_argument("--eps_fx", type=float, default=2.0)
#     # parser.add_argument("--eps_fx", type=str, default='noeps')
#     parser.add_argument("--K", type=int, default=20)
#     parser.add_argument("--text", type=str, default='text')
#     parser.add_argument("--task", type=str, default='20_news')
#     parser.add_argument("--raw_path", type=str, default='output/20_news/data.csv')
#     return parser

# #pubmedqa
def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--eps", type=float, default=2.0)
    parser.add_argument("--eps_fx", type=float, default=2.0)
    # parser.add_argument("--eps_fx", type=str, default='noeps')
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--text", type=str, default='text')
    parser.add_argument("--task", type=str, default='pubmedqa')
    parser.add_argument("--raw_path", type=str, default='output/pubmedqa/data.csv')
    return parser