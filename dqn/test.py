model = "SKT"
number = "72_5"
env_ = "known"

if model == "Vanilla":
    from args import get_args
    from main.agent import Agent
elif model == "SKT" or model == "SKT_super":
    from args_transformer import get_args
    from main.agent_transformer import Agent
elif model == "Vanilla_nopos":
    from args import get_args
    from main.agent_nopos import Agent
elif model == "SKT_nopos" or model == "SKT_nopos_super":
    from args_transformer import get_args
    from main.agent_transformer_nopos import Agent

args = get_args()
args.seed = 222
args.rand_test = True
if env_ == "known":
    args.env_name = 'envs/apps/BoxNav_128_test'
elif env_ == "unknown":
    args.env_name = 'envs/apps/BoxNav_object_test'
else:
    args.env_name = 'envs/apps/ManyManyBoxNav'
args.time_scale = 3
args.device = 'cuda:1'
args.n_test = 5
args.sketch_size = 15

#master_graduation
#eval
args.save_direct = 'results/DQN/master_graduation/{}/'.format(env_)
args.load_dir = 'results/DQN/eval/trained/{}/{}/models/best_model.pt'.format(model, number)

agent = Agent(args)
print(model,"test")
agent.eval_agent(demo=True)
