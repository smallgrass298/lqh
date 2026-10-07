/*=================================================================================
 * The functions of Global
 *=================================================================================*/

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <iostream>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/

#include "Global.hpp"

/*---------------------------------------------------------------------------------
    Name:		Global::SenJud

    Input(1):	The pointer of the sentence

    Function:	Judge the input sentence

    Return:		Return 0 if the sentence is needed;
                Return 1 if the sentence is blank;
                Return 2 if the sentence is the annotate
                ( note if the first char of the sentence is '#',
                  the sentence is taken as an annotate )
 *---------------------------------------------------------------------------------*/

int Global::SenJud(char *sentence)
{
    int i = 0;
    char c = sentence[i];
    if (c == '#')
        return 2;
    if (c == '\n' || c == '\0' || c == ' ')
        return 1;
    else
        return 0;
}

/*---------------------------------------------------------------------------------
    Name:		Global::ParaGet

    Input(3):	The pointer of the sentence;
                The storage of the parameter name;
                The storage of the parameter value

    Function:	Split the sentence into parameter and its value;

    Return:		None
 *---------------------------------------------------------------------------------*/

void Global::ParaGet(char *sentence, char *parameter, char *value)
{
    int i = 0;
    char c = sentence[i];

    while (c != ' ')
    {
        parameter[i] = sentence[i];
        i++;
        c = sentence[i];
    }
    parameter[i] = '\0';

    while (c == ' ')
    {
        i++;
        c = sentence[i];
    }

    int j = 0;
    while (c != '\0')
    {
        value[j] = sentence[i];
        i++;
        j++;
        c = sentence[i];
    }
    value[j] = '\0';
}

int myid;
int numprocs;
int m_block_x, m_block_y, m_block_z;
int m_left = 0, m_right = 0, m_up = 0, m_down = 0, m_front = 0, m_back = 0;
int myid_x, myid_y, myid_z;
int num_thread;