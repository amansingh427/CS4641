{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.7.6"
    },
    "colab": {
      "name": "A1_cs4641b_fall20_student",
      "provenance": [],
      "collapsed_sections": []
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Z6FT4AS4wXDh",
        "colab_type": "text"
      },
      "source": [
        "# Fall CS4641 Assignment 1: Programming\n",
        "\n",
        "## Instructor: Rodrigo Borela\n",
        "\n",
        "## Deadline: September 9, Wednesday, 11:59 pm\n",
        "\n",
        "* No unapproved extension of the deadline is allowed. Late submission will lead to 0 credit. \n",
        "\n",
        "* Discussion on Piazza is encouraged, each student must write their own answers.\n",
        "\n",
        "* The goal of the programming portion of this assignment is to get you familiarized with code submissions to Gradescope and some important Python concepts you will be utilizing throughout the semester. **You are not allowed to use for- or while- loops in any portions of your submission, and all the operations should be executed using Numpy vectorized operations.**\n",
        "\n",
        "* You will copy the appropriate code block indicated in this file and paste it on the file called \"basics.py\", which you will then submit into the assignment named \"A1 Programming\". You can learn more about how to submit your code to Gradescope here: https://help.gradescope.com/article/ccbpppziu9-student-submit-work#submitting_code.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_-Kn8-r9idpV",
        "colab_type": "text"
      },
      "source": [
        "##6 Basic operations using Numpy [15 pts]\n",
        "\n",
        "###6.1 Numpy array broadcasting [5 pts]\n",
        "(Recommended reading: https://numpy.org/doc/stable/user/theory.broadcasting.html#array-broadcasting-in-numpy)\n",
        "\n",
        "Most algorithms we will cover in class involve matrix operations that cannot be implemented efficiently using for- or while- loops. The latter can often be elimitinated employing numpy array broadcasting. In the method \"center_and_scale\" defined under the class \"Basics\" in the following code block, you will modify a dataset $X_{N \\times M}$ such that it has zero mean along its features and the values are scaled within the -1 to 1 range.\n",
        "\n",
        "###6.2 Elementwise and matrix multiplication [3 pts]\n",
        "(Recommended readings: https://numpy.org/doc/stable/reference/generated/numpy.dot.html and https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)\n",
        "\n",
        "When performing multiplications of different kinds involving Numpy arrays it is fundamental to ensure distinction between matrix multiplication and elementwise operations. The character \"*\" used to denote multiplication in Python is limited to elementwise multiplication between arrays following array broadcasting. In the method \"element_vs_matrix\" defined under the class \"Basics\", you will implement both types of operation. Your method should return $Z$, $W$ in which $Z$ is the result of the matrix multiplication $X_{N \\times M}Y_{M \\times P}$, and $W$ is the elementwise multiplication between $X_{N \\times M}$ and $v_M$.\n",
        "\n",
        "###6.3 Numpy indexing [5 pts]\n",
        "(Recommended reading: https://numpy.org/doc/stable/user/basics.indexing.html)\n",
        "\n",
        "Numpy offers simple logical indexing rules that enable you to obtain the indices of values that meet a specific condition without requiring that you loop through each entry in an array and verify the condition is met. In the method \"larger_than\" defined under the class \"Basics\" in the following code block, you will set all entries in matrix $X_{N \\times M}$ larger than a threshold value $t$ to 100.\n",
        "\n",
        "###6.4 Numpy.where [2 pts]\n",
        "(Recommended reading: https://numpy.org/devdocs/reference/generated/numpy.where.html)\n",
        "\n",
        "Similarly to the boolean indexing, you can also perform operations on entries in an array using numpy.where. In the method \"where\" defined under the class \"Basics\" in the following code block, you will return $I,Y$ in which $I$ is the numpy array corresponding to the column indices of all values on the first row that are smaller than the threshold $t$ and $Y_{N \\times M}$ is equivalent to the matrix $X_{N \\times M}$ in which all values smaller than a threshold $t$ are set to -500."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2GpApbKRjpES",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "\"\"\"Copy this code block after you finished implementing the methods and paste it on basics.py.\"\"\"\n",
        "import numpy as np\n",
        "\n",
        "class Basics:\n",
        "    def __init__(self):\n",
        "        pass\n",
        "\n",
        "    # =========================================\n",
        "    # 6.1 Numpy array broadcasting\n",
        "    # =========================================\n",
        "\n",
        "    def center_n_scale(X):\n",
        "        \"\"\"\n",
        "        Input:\n",
        "            X: N x M numpy array, in which the rows correspond to data points and the \n",
        "            columns correspond to features\n",
        "        Output:\n",
        "            Y: N x M numpy array, corresponding to a mean centered and scaled A\n",
        "            array.\n",
        "        \"\"\"\n",
        "        raise NotImplementedError\n",
        "\n",
        "\n",
        "    # =========================================\n",
        "    # 6.2 Elementwise and matrix multiplication\n",
        "    # =========================================\n",
        "\n",
        "    def element_vs_matrix(X, Y, v):\n",
        "        \"\"\"\n",
        "        Input:\n",
        "            X: N x M numpy array\n",
        "            Y: M x P numpy array\n",
        "            v: M dimensional numpy array\n",
        "        Output:\n",
        "            Z: N x P numpy array, corresponding to the matrix multiplication XY\n",
        "            W: N x M numpy array, resulting of the element wise multiplication\n",
        "            between X and v using array broadcasting\n",
        "        \"\"\"\n",
        "        raise NotImplementedError\n",
        "\n",
        "\n",
        "    # =========================================\n",
        "    # 6.3 Numpy indexing\n",
        "    # =========================================\n",
        "\n",
        "    def larger_than(X,t):\n",
        "        \"\"\"\n",
        "        Input:\n",
        "            X: N x M numpy array, containing random integer values\n",
        "            t: integer, threshold value\n",
        "        Output:\n",
        "            Y: N x M numpy array, corresponding to array X in which values larger\n",
        "            than the threshold were set to 100.\n",
        "        \"\"\"\n",
        "        raise NotImplementedError\n",
        "\n",
        "\n",
        "    # =========================================\n",
        "    # 6.4 Numpy.where\n",
        "    # =========================================\n",
        "    \n",
        "    def where(X,t):\n",
        "        \"\"\"\n",
        "        Input:\n",
        "            X: N x M numpy array, containing random integer values\n",
        "            t: integer, threshold value\n",
        "        Output:\n",
        "            I: numpy array, corresponding to indices in the first row of X that\n",
        "            are smaller than the threshold t.\n",
        "            Y: N x M numpy array, corresponding to array X in which values smaller\n",
        "            than the threshold were set to -500.\n",
        "        \"\"\"\n",
        "        raise NotImplementedError"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-Gn0JWNX8DE9",
        "colab_type": "text"
      },
      "source": [
        "**The following code blocks should NOT be copied into basics.py and are here for you to test your code locally**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fyTciOpb0enm",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 202
        },
        "outputId": "bb24aca3-c1f3-481a-b903-25ed23595232"
      },
      "source": [
        "# test center_n_scale\n",
        "A = np.array([[1,17,16,0],[5,13,7,9], [4,3,6,11],[19,1,2,10]])\n",
        "print(\"expected output:\\n\", np.array([[-1.,1.,1.,-1.],[-0.55555556,0.5,-0.28571429,0.63636364],[-0.66666667,-0.75,-0.42857143,1.],[1.,-1.,-1.,0.81818182]]))\n",
        "print(\"======================================================\")\n",
        "print(\"my output:\\n\", Basics.center_n_scale(A))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "expected output:\n",
            " [[-1.          1.          1.         -1.        ]\n",
            " [-0.55555556  0.5        -0.28571429  0.63636364]\n",
            " [-0.66666667 -0.75       -0.42857143  1.        ]\n",
            " [ 1.         -1.         -1.          0.81818182]]\n",
            "======================================================\n",
            "my output:\n",
            " [[-1.          1.          1.         -1.        ]\n",
            " [-0.55555556  0.5        -0.28571429  0.63636364]\n",
            " [-0.66666667 -0.75       -0.42857143  1.        ]\n",
            " [ 1.         -1.         -1.          0.81818182]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "c1o8Bo-r3tdA",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 370
        },
        "outputId": "d95c7392-f7a3-4c03-96e6-a1cef2022420"
      },
      "source": [
        "# test element_vs_matrix\n",
        "A = np.array([[1,17,16,0],[5,13,7,9], [4,3,6,11], [19,1,2,10]])\n",
        "B = np.array([[1,19,10,2,11],[11,6,3,4,14], [5,13,9,7,4], [1,17,16,0,4]])\n",
        "c = np.array([10,2,5,3])\n",
        "Z, W = Basics.element_vs_matrix(A, B, c)\n",
        "print(\"expected Z:\\n\", np.array([[268,329,205,182,313], [192,417,296,111,301],[78,359,279,62,154],[50,563,371,56,271]]))\n",
        "print(\"my output Z:\\n\", Z)\n",
        "print(\"======================================================\")\n",
        "print(\"expected W:\\n\", np.array([[10,34,80,0], [50,26,35,27],[40,6,30,33],[190,2,10,30]]))\n",
        "print(\"my output W:\\n\", W)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "expected Z:\n",
            " [[268 329 205 182 313]\n",
            " [192 417 296 111 301]\n",
            " [ 78 359 279  62 154]\n",
            " [ 50 563 371  56 271]]\n",
            "my output Z:\n",
            " [[268 329 205 182 313]\n",
            " [192 417 296 111 301]\n",
            " [ 78 359 279  62 154]\n",
            " [ 50 563 371  56 271]]\n",
            "======================================================\n",
            "expected W:\n",
            " [[ 10  34  80   0]\n",
            " [ 50  26  35  27]\n",
            " [ 40   6  30  33]\n",
            " [190   2  10  30]]\n",
            "my output W:\n",
            " [[ 10  34  80   0]\n",
            " [ 50  26  35  27]\n",
            " [ 40   6  30  33]\n",
            " [190   2  10  30]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2KXoOfaYeXs2",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 185
        },
        "outputId": "b4ce0597-632e-483f-bc25-12a5690ce62a"
      },
      "source": [
        "# test langer_than\n",
        "A = np.array([[3,25,0,4],[29,2,13,4],[22,27,6,9],[9,10,26,8]])\n",
        "t = 20\n",
        "print(\"expected:\\n\", np.array([[3,100,0,4],[100,2,13,4],[100,100,6,9],[9,10,100,8]]))\n",
        "print(\"my output:\\n\", Basics.larger_than(A,t))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "expected:\n",
            " [[  3 100   0   4]\n",
            " [100   2  13   4]\n",
            " [100 100   6   9]\n",
            " [  9  10 100   8]]\n",
            "my output:\n",
            " [[  3 100   0   4]\n",
            " [100   2  13   4]\n",
            " [100 100   6   9]\n",
            " [  9  10 100   8]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BwJLNaDMxxj6",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 269
        },
        "outputId": "f87ea699-eb60-4ab3-ad5a-c42ec09cd339"
      },
      "source": [
        "# test where\n",
        "A = np.array([[3,25,0,4],[29,2,13,4],[22,27,6,9],[9,10,26,8]])\n",
        "t = 20\n",
        "I, Y = Basics.where(A,t)\n",
        "print(\"expected I:\\n\", np.array([0,2,3]))\n",
        "print(\"my output I: \\n\", I)\n",
        "print(\"======================================================\")\n",
        "print(\"expected Y:\\n\", np.array([[-500,25,-500,-500],[29,-500,-500,-500],[22,27,-500,-500],[-500,-500,26,-500]]))\n",
        "print(\"my output Y: \\n\", Y)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "expected I:\n",
            " [0 2 3]\n",
            "my output I: \n",
            " [0 2 3]\n",
            "======================================================\n",
            "expected Y:\n",
            " [[-500   25 -500 -500]\n",
            " [  29 -500 -500 -500]\n",
            " [  22   27 -500 -500]\n",
            " [-500 -500   26 -500]]\n",
            "my output Y: \n",
            " [[-500   25 -500 -500]\n",
            " [  29 -500 -500 -500]\n",
            " [  22   27 -500 -500]\n",
            " [-500 -500   26 -500]]\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}