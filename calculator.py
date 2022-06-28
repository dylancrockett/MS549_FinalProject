from enum import Enum, unique
import pyparsing as pp
import matplotlib.pyplot as plt
import numpy as np


@unique
class Operator(Enum):
    """
    Operator Enum class with utility methods.
    """
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    EXP = "^"
    EQL = "="

    @classmethod
    def includes(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def get(cls, value):
        return cls._value2member_map_[value]

    def eval(self, right, left):
        """
        Given two values compute the result for the current operator.
        :param left: The first value.
        :param right: The second value.
        :return: The result of using the operand on the two values.
        """
        match self:
            case Operator.ADD:
                return left + right
            case Operator.SUB:
                return left - right
            case Operator.MUL:
                return left * right
            case Operator.DIV:
                return left / right
            case Operator.EXP:
                return pow(left, right)

        raise Exception("Invalid Operator: ", self.value)

    def __int__(self):
        """
        Define int conversion.
        Used for sorting by order of operations.
        :return: The enum as an int.
        """
        return {
            "=": 0,
            "+": 1,
            "-": 1,
            "*": 2,
            "/": 2,
            "^": 3
        }[self.value]

    def __str__(self):
        """
        Define string conversion
        :return: String value of enum.
        """
        return str(self.value)


class CalculatorExpression:
    """
    Expression which represents a computable mathematical expression.
    This includes numbers, symbols, and operators.
    Can contain sub expressions, each expression is scoped in terms of order of operations.
    """
    def __init__(self, calculator: 'Calculator', tokens: list | None = None):
        """
        Expression constructor.
        :param calculator: Instance of calculator object that this expression belongs to.
        """
        self.calc = calculator
        self.tokens = [] if tokens is None else tokens

    def add(self, token: str):
        """
        Add a new token to this expression.

        :param token: The token to be added (as a string)
        :return:
        """
        # check if the token is another expression
        if isinstance(token, CalculatorExpression):
            return self.tokens.append(token)
        # check if the token is an operator
        elif Operator.includes(token):
            return self.tokens.append(Operator.get(token))
        # check if the token is a symbol
        elif token.isalpha():
            # get symbol object
            return self.tokens.append(token)
        # if token is a float
        try:
            return self.tokens.append(float(token))
        except:
            # if token could not be parse
            raise Exception("Invalid expression token.")

    def eval(self) -> float:
        """
        Evaluate the current expression.
        :return: The result of evaluating the expression (a float)
        """
        operand_stack = []
        operator_stack = []

        for token in self.tokens:
            # if operator
            if isinstance(token, Operator):
                while len(operator_stack) > 0 and int(operator_stack[-1]) >= int(token):
                    # ensure there are at least 2 values to compute the value of
                    if len(operand_stack) >= 2:
                        operator = operator_stack.pop()
                        operand_stack.append(operator.eval(operand_stack.pop(), operand_stack.pop()))
                    else:
                        raise Exception("Invalid expression formatting, too many operators.")

                # add operator
                operator_stack.append(token)
            # if expression
            elif isinstance(token, CalculatorExpression):
                # append the evaluated expression
                operand_stack.append(token.eval())
            # if symbol
            elif isinstance(token, str):
                # evaluate symbol and append the value returned
                operand_stack.append(self.calc.symbols[token].eval())
            # if int/float
            elif isinstance(token, int) or isinstance(token, float):
                # append value
                operand_stack.append(token)

        # finish remaining operations
        while len(operator_stack) != 0:
            # ensure there are at least 2 values to compute the value of
            if len(operand_stack) >= 2:
                operator = operator_stack.pop()
                operand_stack.append(operator.eval(operand_stack.pop(), operand_stack.pop()))
            else:
                raise Exception("Invalid expression formatting, too many operators.")

        # if more than one (or no) value is left on the stack then raise an exception
        if len(operand_stack) != 1:
            raise Exception("Invalid expression, too few operators.")

        # return result
        return operand_stack[0]

    def __str__(self):
        """
        Define expression to string conversion
        :return: A string representation of the expression.
        """
        return "(" + " ".join([str(x) for x in self.tokens]) + ")"


class CalculatorSymbolMap:
    """
    A basic hash map which stores symbols and their expressions.

    Map:
        symbol -> CalculatorExpression
    """
    def __init__(self):
        # hash map
        self.map = {}

    def __setitem__(self, symbol: str, expr: CalculatorExpression):
        """
        Add support for adding using [] syntax.
        :param symbol: The symbol.
        :param expr: The symbol's expression.
        :return:
        """
        self.add(symbol, expr)

    def __getitem__(self, symbol: str):
        """
        Add support for getting using [] syntax.
        :param symbol: The symbol.
        :return: The symbol's expression.
        """
        return self.get(symbol)

    def add(self, symbol: str, expr: CalculatorExpression) -> None:
        """
        Add a new symbol definition.
        :param symbol: The symbol.
        :param expr: The symbol's expression.
        :return:
        """
        self.map[symbol] = expr

    def get(self, symbol: str) -> CalculatorExpression:
        """
        Get the expression for a given symbol.
        :param symbol: The symbol.
        :return: The symbol's expression.
        """
        return self.map[symbol]

    def items(self):
        return self.map.items()

    def __len__(self):
        return len(self.map.keys())


class Calculator:
    """
    Calculator session instance.

    Handles storing symbol definitions over lifetime.
    Can do computations using symbol definitions.
    """
    def __init__(self):
        """
        Calculator constructor.
        """
        # create symbol map instance
        self.symbols = CalculatorSymbolMap()

    def eval(self, expr_str: str) -> str:
        """
        Evaluate a given expression, handles adding new symbols and some other basic calculator commands.
        :param expr_str: The expression being evaluated.
        :return:
        """
        # if empty string
        if expr_str == "":
            return "Cannot parse empty expression."

        try:
            # parse the string expression
            expression = self.parse(expr_str)

            # commands
            if len(expression.tokens) >= 1 and isinstance(expression.tokens[0], str):
                if expression.tokens[0].lower() == "graph":
                    if len(expression.tokens) == 3:
                        # render graph
                        self.graph(expression.tokens[1], expression.tokens[2])
                        return "Graph rendered."
                    else:
                        return "Graph command requires 2 arguments"
                if expression.tokens[0].lower() == "symbols":
                    if len(self.symbols) == 0:
                        return "There are currently no defined symbols."

                    return "Currently Defined Symbols:\n" \
                           + "\n".join([f"{s} = {str(expr)}" for (s, expr) in self.symbols.items()])

            # if the expression is setting a symbol
            if len(expression.tokens) > 2 and isinstance(expression.tokens[0], str) \
                    and expression.tokens[1] == Operator.EQL:
                # pop first two tokens
                symbol = expression.tokens.pop(0)
                expression.tokens.pop(0)

                # map the new symbol
                self.symbols[symbol] = expression

                # return message
                return f"Added new symbol '{symbol}' with a value of '{expression}'"
            else:
                # return evaluated expression
                return expression.eval()
        except KeyError as e:
            return "Symbol '" + str(e) + "' has not been defined."
        except Exception as e:
            return str(e)

    @staticmethod
    def tokenize(expr_str: str) -> pp.ParseResults:
        """
        Tokenize a given expression in string form.
        :param expr_str: The string to be tokenized.
        :return: The parsing results.
        """
        # create tokenize instance
        tokenizer = pp.Forward()

        # define token sub-types
        symbol_token = pp.Word(pp.alphas)
        number_token = pp.Combine(pp.Optional("-") + pp.Word(pp.nums) + pp.Optional(pp.Combine("." + pp.Word(pp.nums))))
        operator_token = pp.Char("+-*/^=")

        # define token type
        token = pp.OneOrMore(symbol_token ^ number_token ^ operator_token)

        # define sub expression definition
        expression_token = pp.nestedExpr(opener="(", closer=")", content=token)

        # define tokenizer
        tokenizer << pp.OneOrMore(expression_token ^ token)

        # return parse results
        return tokenizer.parse_string(expr_str, parse_all=True)

    def parse(self, expr_str: str | pp.ParseResults) -> CalculatorExpression:
        """
        Parse a given string expression into CalculatorExpression format.
        :param expr_str: The string to be parsed.
        :return: The resulting CalculatorExpression instance.
        """
        # tokenize input expression if needed
        tokens = expr_str if isinstance(expr_str, pp.ParseResults) else self.tokenize(expr_str)

        # create expression instance
        expression = CalculatorExpression(self)

        # go through each token
        for token in tokens:
            # check if it is a sub expression
            if isinstance(token, pp.ParseResults):
                # convert sub expression to CalculatorExpression
                sub = self.parse(token)

                # add it to main expression
                expression.add(sub)
            else:
                # add other token directly
                expression.add(token)

        # return finished expression
        return expression

    def graph(self, output_symbol: str, input_symbol: str):
        # 100 linearly spaced numbers
        x_vals = np.linspace(-5, 5, 100)

        # create sub calculator instance
        sub_calc = Calculator()
        sub_calc.symbols = self.symbols

        # the function, which is y = x^2 here
        def f(x):
            sub_calc.symbols[input_symbol] = CalculatorExpression(sub_calc, [x])
            return sub_calc.symbols[output_symbol].eval()

        f_vec = np.vectorize(f)
        y_vals = f_vec(x_vals)

        # setting the axes at the centre
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # plot the function
        plt.plot(x_vals, y_vals, 'r')
        plt.show()
