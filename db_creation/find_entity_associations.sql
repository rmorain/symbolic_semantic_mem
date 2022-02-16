---Get all the associations where a specific entity is the related entity
SELECT relations.entity_label, relations.property_label, Entities.label as related_entity_label
FROM Entities
JOIN (
SELECT e.entity_id, e.label AS entity_label, p.property_id, p.label AS property_label, pr.related_entity_id
FROM Properties_relations as pr
LEFT JOIN Entities as e
ON pr.entity_id = e.entity_id
LEFT JOIN Properties as p
ON pr.property_id = p.property_id
WHERE pr.related_entity_id = "Q31"
)
AS relations
ON relations.related_entity_id = Entities.entity_id
LIMIT 100;
