import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default='100', help="IP address")
    parser.add_argument('--seed', type=int, default=345)
    parser.add_argument(
        "--model",
        help="choose model",
        type=str,
        choices=[
            "SKT",
            "CNN",
            "AQT",
        ],
    )
    parser.add_argument('--n-epochs', type=int, default=1000)
    parser.add_argument('--n-cycles', type=int, default=50) # sample-update per epoch
    parser.add_argument('--n-updates', type=int, default=100) # updates per cycle
    parser.add_argument('--ep-len', type=int, default=100)
    parser.add_argument('--n-test', type=int, default=10)
    parser.add_argument('--action-size', type=int, default=4)
    parser.add_argument('--sketch-size', type=int, default=1)
    parser.add_argument('--epsilon-start', type=float, default=0.99)
    parser.add_argument('--epsilon-end', type=float, default=0.01)
    parser.add_argument('--epsilon-decay', type=int, default=2000)
    parser.add_argument('--start-train', type=int, default=100)
    parser.add_argument('--buffer-size', type=int, default=250000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--replay-frequency', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--env-name', type=str, default='envs/apps/Indoor_bird')
    parser.add_argument('--save-direct', type=str, default='results/DQN/')
    parser.add_argument('--obs-shape', type=list, default=(3,128,128))
    parser.add_argument('--time-scale', type=float, default=20.0)
    parser.add_argument('--worker-id', type=int, default=0)
    parser.add_argument('--alpha-start', type=float, default=1.0)
    parser.add_argument('--alpha-end', type=float, default=0.0)
    parser.add_argument('--alpha-decay', type=int, default=2500)
    parser.add_argument('--super-model', type=int, default=5)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--rand-test', action="store_true", default=True)
    args = parser.parse_args()#args=[]
    return args

def main():
    args = get_args()
    args.device = "cuda:{}".format(args.device)
    if args.model == "SKT":
        from main.agent_transformer import Agent
    elif args.model == "AQT":
        from main.agent_act_q_transformer import Agent
    else:
        from main.agent import Agent
    agent = Agent(args)
    agent.learn()

if __name__ == "__main__":
    main()