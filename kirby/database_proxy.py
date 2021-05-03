# AUTOGENERATED! DO NOT EDIT! File to edit: 08_database_proxy.ipynb (unless otherwise specified).

__all__ = ['WikiDatabase']

# Cell
import pandas as pd
import sqlite3
import time
from .properties import properties

# Cell
class WikiDatabase:
    conn = None
    def __init__(self):
        try:
            self.properties_dict = properties()
            self.conn = sqlite3.connect("/data/db/wiki_data.db")
            print(self.conn)
        except Exception as e:
            print(e)
            exit(-1)

    def exit_procedure(self):
        pass
#         self.conn.close()
#         exit(-1)


    @staticmethod
    def get_table_name(entity_label):
        """
        @:param self:
        @:param label of the entity for which you need to
        :rtype: string
        @:return name of the table where entity is stored
        """
        if entity_label[0].isalpha():
            return entity_label[0].lower()
        else:
            return 'not_alpha'

    @staticmethod
    def clean_relations(relations_string):
        """
        :param relations_string: List of all the relations returned by database
        :type relations_string: string
        :return: list of all relations
        :rtype: List
        """
        relations = []
        relation_row = []
        # print(relations_string)
        for idx, relation in enumerate(relations_string.split(',')):
            if idx & 1 == 1:
                relation_row.append(relation[1:-1])
            else:
                relation_row.append(relation[1:])
            if len(relation_row) == 2:
                relations.append(relation_row + [])
                # print(relation_row)
                relation_row.clear()
        # print(relations)
        return relations

    @staticmethod
    def get_property_label(relations_id):
        """
            :param property_id: id of the property to look for
            :type property_id:
            :return: label of the given id, None if the id does not exist
            :rtype:
        """
        return property_dict[relations_id]

    @staticmethod
    def remove_quotations(label):
        index = 0
        while index < len(label):
            if label[index] == '\'':
                label = label[:index] + '\'' + label[index:]
                index += 1
            elif label[index] == "\"":
                label = label[:index - 1] + '\'' + '\'' + label[index:]
                index += 1
            index += 1

        return label

    # TODO: Test the return value
    # TODO: change to check in the entities table
    def get_similar_entities_label(self, label):
        """
        Returns a list of entities id where the label matches the entity_label name
        :param self:
        :type self:
        :param label:
        :type label:
        :return: list of all the entities id and label where the label matched
        :rtype: Array containing all similar entities
        """
        entities_id = None
        try:
            print('I love me!')
            entities_id = pd.read_sql_query(
                "SELECT * FROM Entities WHERE label  LIKE \"%{}%\";".format(self.remove_quotations(label)),
                self.conn)
        except Exception as e:
            print(e)
            self.exit_procedure()
        return entities_id.values.tolist()

    def get_entities_by_label_extensive(self, label):
        """
        Returns a list of entities id where the label matches the entity_label name
        :param self:
        :type self:
        :param label:
        :type label:
        :return: list of all the entities id and label where the label matched
        :rtype: Array containing all similar entities
        """
        table_name = self.get_table_name(label)
        entities_id = None
        try:
            entities_id = pd.read_sql_query(
                "SELECT * FROM Entities_{} WHERE label  LIKE \"{}%\";"\
                .format(table_name, self.remove_quotations(label)), self.conn)
            return entities_id.values.tolist()
        except Exception as e:
            print(e)
            self.exit_procedure()


    def get_entity_by_label(self, label):
        """
        Search for the id, label and description of an specific identity
        :param label: string of the label to look for
        :type label: string
        :return: id, label and description of the label
        :rtype: pandas.core.series.Series
        """
        entity_id = None
        table_name = self.get_table_name(label)
        try:
            entity_id = pd.read_sql_query(
                "SELECT * FROM Entities_{} WHERE label =  \"{}\";"\
                .format(table_name, self.remove_quotations(label)), self.conn)
            entities = entity_id.values.tolist()
            # Return the first entity
            return entities[0]
        except Exception as e:
            print(e)
            self.exit_procedure()

    def get_entity_associations(self, entity_id):
        """
        Search for all properties of the entity
        Return all the
        :param entity_id: id of a entity
        :type entity_id: str
        :return: data frame with a string of the relations
        :rtype: pandas.core.series.Series
        """
        entity_properties = None
        try:
            entity_properties = pd.read_sql_query(
                "SELECT * FROM Properties_relations WHERE entity_id = \"{}\";".format(entity_id),
                self.conn)
        except Exception as e:
            print(e)
            self.exit_procedure()
        if entity_properties.empty:
            return None
        return self.clean_relations(entity_properties['relations'].values[0])

    def get_entity_by_id(self, entity_id):

        relation_label = None
        try:
            relation_label = pd.read_sql_query(
                "SELECT label, description FROM Entities WHERE entity_id = \"{}\"".format(entity_id),
                self.conn)
        except Exception as e:
            print(e)
            self.exit_procedure()
        if relation_label.empty:
            return None
        return relation_label.iloc[0]["label"]

    def get_property_string(self, property_id, related_entity_id):
        """
        :param property_id:
        :type property_id: string
        :param related_entity_id:
        :type related_entity_id: string
        :return:
        :rtype: string describing the relationship
        """
        entity_label = None
        related_entity_label = None
        property_name = None
        try:
            # entity_label = get_entity_label(entity_id)
            related_entity_label = self.get_entity_by_id(related_entity_id)
            property_name = self.properties_dict[property_id]
            # print(related_entity_label, property_name)
        except Exception as e:
            print(e)
            self.exit_procedure()
        return property_name, related_entity_label